import multiprocessing


def null_func(npart, **kwargs):
    pass


def multi_proc(npart, pre_proc=null_func, part_job=null_func, post_proc=null_func, **kwargs):
    pool = multiprocessing.Pool(processes=npart)
    jobs = pre_proc(npart, **kwargs)
    results = []
    for y in pool.imap_unordered(part_job, jobs):
        results.append(y)
    kwargs['results'] = results
    return post_proc(npart, **kwargs)
