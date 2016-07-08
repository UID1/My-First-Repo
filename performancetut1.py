def perf_comp_data(func_list, data_list, rep=3, number=1):
    """
    Function to compare the performance of different functions.

    Parameters
    ==========
    func_list : list
        list with function names as strings
    data_list : list
        list with data set names as strings
    rep : int
        number of executions for every function
    """
    from timeit import repeat
    res_list = {}
    for name in enumerate (func_list):
        stmt = name[1] + '(' + data_list[name[0]] + ')'
        setup = "from __main__ import " + name[1] + ', ' \
                    + data_list[name[0]]
        results = repeat(stmt=stmt, setup=setup,
                            repeat=rep, number=number)
        res_list[name[1]] = sum(results) / rep
    res_sort = sorted(res_list.iteritems(),
                        key=lambda (k, v): (v, k))
    for item in res_sort:
        rel = item[1] / res_sort[0][1]
        print 'function: ' + item[0] + \
                ', av. time sec: %9.5f, ' %item[1] \
                + 'relative : %6.1f' % rel
