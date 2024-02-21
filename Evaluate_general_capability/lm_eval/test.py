import collections
import itertools
import random

import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import numpy as np
import transformers

model = 'hf-causal'
tasks = ['boolq','arc_easy']
model_args = 'pretrained="/data/wklu/LLMs/Llama2-hf/EarlyStop_once_1_1percent/"'
batch_size = 4
max_batch_size = None
device = 'cuda:7'
description_dict = {}
num_fewshot=0
decontaminate_suffix = "_decontaminate"
bootstrap_iters=100000
random.seed(1234)
np.random.seed(1234)

assert tasks != [], "No tasks specified"

if isinstance(model, str):
    if model_args is None:
        model_args = ""
    lm = lm_eval.models.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
            "device": device,
        },
    )
elif isinstance(model, transformers.PreTrainedModel):
    lm = lm_eval.models.get_model("hf-causal")(
        pretrained=model,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )
    no_cache = True
else:
    assert isinstance(model, lm_eval.base.LM)
    lm = model

if not no_cache:
    lm = lm_eval.base.CachingLM(
        lm,
        "lm_cache/"
        + (model if isinstance(model, str) else model.model.config._name_or_path)
        + "_"
        + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
        + ".db",
    )

task_dict = lm_eval.tasks.get_task_dict(tasks)

task_dict_items = [
    (name, task)
    for name, task in task_dict.items()
    if (task.has_validation_docs() or task.has_test_docs())
]

results = collections.defaultdict(dict)
versions = collections.defaultdict(dict)

requests = collections.defaultdict(list)
requests_origin = collections.defaultdict(list)

overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

# If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
# memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
# over-engineering is bad (or we could make it write the requests to disk and then read them back out again
#  - probably using an sqlite db because of all the moving parts we have

# TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
docs = {}
write_out_info = {}

docs_for_decontamination = collections.defaultdict(list)

# get lists of each type of request
for task_name, task in task_dict_items:
    versions[task_name] = task.VERSION
    # default to test doc, fall back to val doc if validation unavailable
    # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
    if task.has_test_docs():
        task_doc_func = task.test_docs
        task_set = "test"  # Required for caching in the decontamination
    elif task.has_validation_docs():
        task_set = "val"  # Required for caching in the decontamination
        task_doc_func = task.validation_docs
    else:
        raise RuntimeError("Task has neither test_docs nor validation_docs")

    # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
    task_docs = list(task_doc_func())
    rnd = random.Random()
    rnd.seed(42)
    rnd.shuffle(task_docs)
    print(f"Task: {task_name}; number of docs: {len(task_docs)}")

    description = (
        description_dict[task_name]
        if description_dict and task_name in description_dict
        else ""
    )
    if limit is not None:
        limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

    for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):

        docs[(task_name, doc_id)] = doc
        ctx = task.fewshot_context(
            doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
        )
        reqs = task.construct_requests(doc, ctx)

        # print the prompt for the first few documents
        if doc_id < 1:
            print(
                f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
            )
            print("Requests:", reqs)

        if not isinstance(reqs, (list, tuple)):
            reqs = [reqs]
        for i, req in enumerate(reqs):
            requests[req.request_type].append(req)
            # i: index in requests for a single task instance
            # doc_id: unique id that we can get back to a doc using `docs`
            requests_origin[req.request_type].append((i, task_name, doc, doc_id))

# Compare all tasks/sets at once to ensure a single training set scan


# all responses for each (task, doc)
process_res_queue = collections.defaultdict(list)

# execute each type of request
for reqtype, reqs in requests.items():
    # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
    #       only in index. We could implement some kind of caching, but that would be more of a band-aid
    #       solution. we could also implement some kind of auto-grouping here;
    #       they should end up next to each other.

    print("Running", reqtype, "requests")
    resps = getattr(lm, reqtype)([req.args for req in reqs])
    resps = [
        x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
    ]

    for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
        process_res_queue[(task_name, doc_id)].append((i, resp))

vals = collections.defaultdict(list)

# unpack results and sort back in order and return control to Task
for (task_name, doc_id), requests in process_res_queue.items():
    requests.sort(key=lambda x: x[0])
    requests = [x[1] for x in requests]

    task = task_dict[task_name]
    doc = docs[(task_name, doc_id)]

    metrics = task.process_results(doc, requests)
    for metric, value in metrics.items():
        vals[(task_name, metric)].append(value)


# aggregate results
for (task_name, metric), items in vals.items():
    task = task_dict[task_name]
    real_metric = metric  # key when looking up the metric with task.aggregation
    if metric.endswith(decontaminate_suffix):
        real_metric = metric.replace(
            decontaminate_suffix, ""
        )  # decontaminated still uses the same metric
    results[task_name][metric] = task.aggregation()[real_metric](items)

    # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
    # so we run them less iterations. still looking for a cleaner way to do this

    stderr = lm_eval.metrics.stderr_for_metric(
        metric=task.aggregation()[real_metric],
        bootstrap_iters=min(bootstrap_iters, 1000)
        if metric in ["bleu", "chrf", "ter"]
        else bootstrap_iters,
    )

    if stderr is not None:
        results[task_name][metric + "_stderr"] = stderr(items)
