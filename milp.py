import math
import json
from gurobipy import *

from visualize import Visualization

# try:
# PRECISION
EPSILON = 0.0001

SLOT_LENGTH = 10

# nodes with root
N = None
# nodes without root
N_0 = None
# nodes mapped to parent - dictionary
P = None
# nodes mapped to their children - dictionary
C = None
# slots in a slot frame
# maximum slot in a slotframe
T_MAX = None
# goes from 0 to T_MAX
T = None
# # the minimal SHARED cells used for bootstrapping and management
# T_MINIMAL = None
# number of times an aggregated slot can be repeated in a slotframe
R = None
# number of slots an aggregated slot can contain
S = None
# frequencies at which a node can transmit
# maximal frequency offset
F_MAX = None
# goes from 0 to F_MAX
F = None
# tranmission rate: number of packets generated per slot frame
# assumption, set to 1
x_n = 1
# dictionary 'Rel' of all possible links with the MCSs, mapped to the corresponding reliability
Rel = None
# set of modulation and coding schemes (MCSs) M
M = []
# dictionary of MCS to the number of slots necessary in an aggregated slot
SDict = None

# A = {
#     'QAM_16_FEC_3_4': 4.11,
#     'QAM_16_FEC_1_2': 6.16,
#     'QPSK_FEC_3_4': 8.21,
#     'QPSK_FEC_1_2': 12.32,
#     'QPSK_FEC_1_2_FR_2': 24.64,
# }

A = {
    'QAM_16_FEC_3_4': 6.03,
    'QAM_16_FEC_1_2': 8.08,
    'QPSK_FEC_3_4': 10.13,
    'QPSK_FEC_1_2': 14.24,
    'QPSK_FEC_1_2_FR_2': 26.56,
}


omega = 1.0  # the weight
gap = 0
threads = 1
# timelimit = 82800 # 23 hours
# timelimit = 3000 # 50 minutes
timelimit = 0 # we interpret 0 as no time limit

### INPUTS
if len(sys.argv) > 1:
    # parse the simulation topology
    json_file = sys.argv[1]
    if len(sys.argv) <= 2:
        assert False # there should be a pathname for the schedule file!
    exp_file = None
    with open(json_file) as data_file:
        exp_file = json.load(data_file)
    if 'simulationTopology' not in exp_file:
        raise RuntimeError('No simulation topology in the exp_file feeded to the ILP.')
    topology = exp_file['simulationTopology']

    omega = float(exp_file['omega'])
    assert 0.0 <= omega <= 1.0
    gap = float(exp_file['gap'])
    assert 0.0 <= gap <= 1.0
    threads = int(exp_file['threads'])
    assert 1 <= threads
    timelimit = int(exp_file['timelimit'])
    assert 0 <= timelimit
    SLOT_LENGTH = int(exp_file['slotLength'])
    assert 0 < SLOT_LENGTH <= 30

    # set of modulation and coding schemes (MCSs) M
    M = []
    # dictionary 'Rel' of all possible links with the MCSs, mapped to the corresponding reliability
    Rel = {}
    N = []
    N_0 = []
    P = {}
    C = {}

    for tmp_n, value in topology.iteritems():
        n = int(tmp_n)
        N.append(n)

        # without the root
        if n != 0:
            # add the nodes without the root
            N_0.append(n)
            # add the parents
            P[n] = int(value['parent'])
            # add the children
            if int(value['parent']) not in C:
                C[int(value['parent'])] = []
            C[int(value['parent'])].append(n)

            # go over all possible MCSs, but only save the values towards the parent
            for reliability_mcs, reliability_dict in value['reliability'].iteritems():
                # set the reliability from node to parent for a given mcs
                Rel[(n, P[n], reliability_mcs)] = reliability_dict[str(P[n])]
                # make sure you have a list of all the MCSs
                if reliability_mcs not in M:
                    M.append(reliability_mcs)

    # print M
    # assert False

    # sort them to have nice visualization
    N.sort()
    N_0.sort()

    # dictionary of MCS to the number of slots necessary in an aggregated slot
    SDict = exp_file['modulationSlots']
    max_s = 0
    for (m, s) in SDict.iteritems():
        if s > max_s:
            max_s = s

    # slots in a slot frame
    T_MAX = exp_file['slotframeLength'] - 1
    T = range(0, T_MAX + 1)  # goes from 0 to T_MAX
    # T_MINIMAL = []
    # for (ts, ch, modulation) in exp_file['minimal_slots']:
    #     assert modulation in SDict
    #     for i in range(SDict[modulation]):
    #         T_MINIMAL.append(ts + i)
    # print T_MINIMAL

    # number of times an aggregated slot can be repeated in a slotframe
    # R = range(1, T_MAX + 2)  # goes from 1 to T_MAX + 1
    R = range(1, T_MAX + 2)  # goes from 1 to T_MAX + 1
    if T_MAX + 2 > 5:
        R = range(1, 5)  # goes from 1 to 5

    # number of slots an aggregated slot can contain
    # S = range(1, T_MAX + 2)  # goes from 1 to T_MAX + 1
    S = range(1, max_s + 1)  # goes from 1 to T_MAX + 1

    # frequencies at which a node can transmit
    F_MAX = exp_file['numChans'] - 1  # maximal frequency offset
    F = range(0, F_MAX + 1)  # goes from 0 to F_MAX

    # print N
    # print N_0
    # print P
    # print C
    # print SDict
    # print Rel
    # print R
    # print S
    # print S
else:
    # nodes in a network
    # N = [0, 1, 3]
    # N = [0, 1, 2, 3, 4]
    N = [0, 1, 2, 3]
    N_0 = [1, 2, 3]

    # dictionary map from node to parent of the node
    P = dict({
        1: 0,
        2: 1,
        3: 2,
        # 4: 3
        # 4: 2
    })

    # dictionary map from node to its children
    C = dict({
        0: [1],
        1: [2],
        2: [3],
        # 3: [4],
    })

    # slots in a slot frame
    T_MAX = 6 # maximum slot in a slotframe
    T = range(0, T_MAX + 1)  # goes from 0 to T_MAX

    # number of times an aggregated slot can be repeated in a slotframe
    R = range(1, T_MAX + 2)  # goes from 1 to T_MAX + 1
    if T_MAX + 2 > 5:
        R = range(1, 5)  # goes from 1 to 5

    # number of slots an aggregated slot can contain
    S = range(1, T_MAX + 2)  # goes from 1 to T_MAX + 1

    # frequencies at which a node can transmit
    F_MAX = 1  # maximal frequency offset
    # F_MAX = 2 # maximal frequency offset
    F = range(0, F_MAX + 1)  # goes from 0 to F_MAX

    # set of modulation and coding schemes (MCSs) M
    # dictionary of MCS to the number of slots necessary in an aggregated slot
    SDict = dict({
        # 'QAM_16_FEC_3_4': 2,
        # 'QAM_16_FEC_1_2': 1,
        # 'QPSK_FEC_3_4': 2,
        'QPSK_FEC_1_2': 3,
    })

    M = ['QPSK_FEC_1_2']

    # dictionary 'Rel' of all possible links with the MCSs, mapped to the corresponding reliability
    Rel = dict({
        # (1, 0, 'QAM_16_FEC_1_2'): 1,
        # (2, 1, 'QAM_16_FEC_1_2'): 1,
        # (1, 0, 'QAM_16_FEC_3_4'): 1,
        # (2, 0, 'QAM_16_FEC_3_4'): 1,
        # (3, 1, 'QPSK_FEC_1_2'): 1.0,
        (1, 0, 'QPSK_FEC_1_2'): 0.4,
        # (1, 0, 'QPSK_FEC_3_4'): 1.0,
        # (3, 1, 'QAM_16_FEC_1_2'): 1.0,
        (2, 1, 'QPSK_FEC_1_2'): 0.6,
        # (4, 2, 'QPSK_FEC_1_2'): 1.0,
        # (2, 1, 'QPSK_FEC_3_4'): 1.0,
        # (2, 0, 'QPSK_FEC_3_4'): 0.5,
        # (3, 2, 'QPSK_FEC_1_2'): 1,
        (3, 2, 'QPSK_FEC_1_2'): 0.9,
        # (4, 3, 'QPSK_FEC_1_2'): 0.9,
        # (1, 0, 'QPSK_FEC_3_4'): 1,
        # (2, 1, 'QPSK_FEC_3_4'): 0.75,
    })


### AUXILIARY SYMBOLS

def fPDR(n, m, r, x_n_tmp):
    if r == 1:
        return x_n_tmp * Rel[(n, P[n], m)]
    else:
        return fPDR(n, m, r - 1, x_n_tmp) + (1 - fPDR(n, m, r - 1, x_n_tmp)) * Rel[(n, P[n], m)]

def getAllChildren(n):
    if n not in C:
        return []
    else:
        all = []
        for child in C[n]:
            all += [child]
            all += getAllChildren(child)
        return all

# Create our 'Adaptive TSCH' optimization model
m = Model('adaptsch')

### DECISION VARIABLES

# binary var that is 1 when node n has s consecutive time slots allocated at time offsets t, t + 1, ..., t + s - 1
# and frequency offset f in the TSCH schedule, to its parent
sigma = m.addVars(T, F, N_0, S, vtype=GRB.BINARY, name='sigma')

# binary var that is 1 when node n uses MCS m and r repetitions of the aggregated slot to its parent
beta = m.addVars(R, M, N_0, vtype=GRB.BINARY, name='beta')

# integer decision variable that makes sure that the number of packets a node tries to send is an integer
delta = m.addVars(N_0, vtype=GRB.INTEGER, name='delta')

arr = m.addVars(N_0, vtype=GRB.CONTINUOUS, name='arr')


### CONSTRAINTS

# equation 2
m.addConstrs((beta.sum('*', '*', n) == 1 for n in N_0), name='eq_2')

# equation 3
for n in N_0:
    for t in T:
        constr = LinExpr()
        for f in F:
            for _t in range(t + 1):
                s_min_max = min(max(S) + 1, (T_MAX + 1) - _t + 1)
                for s in range(t - _t + 1, s_min_max):
                    constr += sigma[_t, f, n, s]  # the constraint for the node itself
                    if n in C:  # if it has children
                        for j in C[n]:  # account for each child
                            constr += sigma[_t, f, j, s]
        m.addConstr((constr <= 1), name='eq_3[%s,%s]' % (n, t))

# equation 4
for t in T:
    constr = LinExpr()
    for f in F:
        for _t in range(t + 1):
            s_min_max = min(max(S) + 1, (T_MAX + 1) - _t + 1)
            for s in range(t - _t + 1, s_min_max):
                if 0 in C:  # if it has children
                    for j in C[0]:  # account for each child
                        constr += sigma[_t, f, j, s]
    m.addConstr((constr <= 1), name='eq_4[%s,%s]' % (0, t))


# equation 5
for n in N_0:
    for t in T:
        for f in F:
            for s in S:
                m.addConstr((sigma[t, f, n, s] * (t + s - 1) <= T_MAX), name='eq_5[%s,%s,%s,%s]' % (t, f, n, s))

# equation 6
for n in N_0:
    for t in T:
        for f in F:
            for mcs in M:
                for r in R:
                    for s in S:
                        if s != SDict[mcs]:
                            m.addConstr((beta[r, mcs, n] * sigma[t, f, n, s]) == 0, name='eq_6[%s,%s,%s,%s,%s,%s]' % (n, t, f, mcs, r, s))

# equation 7
for n in N_0:
    rhsConstr = LinExpr()
    for mcs in M:
        for r in R:
            constrTmp = LinExpr()
            for t in T:
                for f in F:
                    for s in S:
                        constrTmp += sigma[t, f, n, s]
            rhsConstr += (constrTmp / float(r)) * beta[r, mcs, n]
    m.addConstr(delta[n] == rhsConstr, name='eq_7[%s]' % (n))

# equation 8
for n in N_0:
    lhsConstr = LinExpr()
    lhsConstr += delta[n] - x_n

    rhsConstr = LinExpr()
    if n in C:
        for child in C[n]:
            rhsConstr += arr[child]
    rhsConstr += (1 - EPSILON)

    m.addConstr(lhsConstr <= rhsConstr, name='eq_8[%s]' % (n))

# equation 9
for n in N_0:
    rhsConstr = LinExpr()
    for mcs in M:
        for r in R:
            rhsConstr += (beta[r, mcs, n] * fPDR(n, mcs, r, 1))

    x_n_and_arr_child = LinExpr()
    x_n_and_arr_child += x_n
    if n in C:
        for child in C[n]:
            x_n_and_arr_child += arr[child]

    rhsConstr *= x_n_and_arr_child

    m.addConstr(arr[n] <= rhsConstr, name='eq_9[%s]' % (n))

# equation 10
for n in N_0:
    rhsConstr = LinExpr()
    for mcs in M:
        for r in R:
            rhsConstr += beta[r, mcs, n] * fPDR(n, mcs, r, 1)
    rhsConstr *= delta[n]

    m.addConstr(arr[n] <= rhsConstr, name='eq_10[%s]' % (n))

### OBJECTIVE

m.modelSense = GRB.MAXIMIZE

if timelimit > 0:
    m.setParam('TimeLimit', timelimit)
if gap > 0.0:
    m.setParam('MIPGap', gap)
if threads > 1:
    m.setParam('Threads', threads)
else:
    m.setParam('Threads', 1)

# minimize the total number of slots
radioTime = LinExpr()
for n in N_0:
    sigmas = LinExpr()
    for t in T:
        for f in F:
            for s in S:
                sigmas += sigma[t, f, n, s]

    beta_As = LinExpr()
    for mcs in M:
        for r in R:
            beta_As += beta[r, mcs, n] * A[mcs]
    radioTime += (sigmas * beta_As)

arr_root = LinExpr()
for child in C[0]:
    arr_root += arr[child]

of = (omega * (arr_root / float(sum([x_n for n in N_0]))) - (1 - omega) * (radioTime / float(len(T) * SLOT_LENGTH * len(N_0))))

m.setObjective(of)

# Save the problem
m.write('adaptsch.lp')

# Optimize
m.optimize()

status = m.Status
if status == GRB.Status.INF_OR_UNBD or \
        status == GRB.Status.INFEASIBLE or \
        status == GRB.Status.UNBOUNDED:
    raise RuntimeError('The model cannot be solved because it is infeasible or unbounded')
    sys.exit(0)

if status == GRB.Status.OPTIMAL or status == GRB.Status.TIME_LIMIT:
    print('There are %s solutions found.' % m.SOLCOUNT)

    solution_sigma = m.getAttr('x', sigma)
    solution_beta = m.getAttr('x', beta)
    solution_delta = m.getAttr('x', delta)
    solution_arr = m.getAttr('x', arr)

    # CHECKS

    TEST_ILP = True

    # if TEST_ILP:
    #     # should conflict with SIGMA[0, 1, 1, 1] at another frequency
    #     # solution_sigma[0, 2, 1, 1] = 1
    #     # should conflict with SIGMA[0, 1, 1, 1] at another s
    #     # solution_sigma[0, 2, 2, 3] = 1
    #     # solution_sigma[3, 2, 2, 1] = 1
    #     solution_beta[1, 'QAM_16_FEC_1_2', 1] = 0
    #     solution_beta[2, 'QAM_16_FEC_1_2', 1] = 1
    #     # solution_beta[3, 'QAM_16_FEC_3_4', 1] = 1
    # solution_

    # solution_sigma[3, 1, 1, 2] = 1
    # solution_sigma[3, 0, 2, 2] = 1
    # solution_sigma[1, 0, 2, 1] = 1
    # solution_beta[1, 'QAM_16_FEC_1_2', 3] = 1


    # VISUALIZATION

    viz = Visualization(len(T), len(F), N, P, omega)

    # objective function

    ## the arr_c of all children of the root

    arr_root_list = []
    for child in C[0]:
        if solution_arr[child] > EPSILON:
            arr_root_list.append((child, solution_arr[child]))
    viz.add_arr_root_list(arr_root_list)

    x_list = []
    for child in N_0:
        x_list.append((child, x_n))
    viz.add_x_list(x_list)

    total_arr = 0.0
    for (c, arr_child) in arr_root_list:
        total_arr += arr_child
    total_x = 0.0
    for (c, x_child) in x_list:
        total_x += x_child
    calculated_throughput = (total_arr / float(total_x))

    ## the total number of slots

    total_airtime = 0
    for n in N_0:
        all_sigmas = 0.0
        for t in T:
            for s in S:
                for f in F:
                    if solution_sigma[t, f, n, s] > EPSILON:
                        all_sigmas += solution_sigma[t, f, n, s]
        for mcs in M:
            for r in R:
                if solution_beta[r, mcs, n] > EPSILON:
                    total_airtime += (all_sigmas * A[mcs])

    viz.add_total_slots(total_airtime)
    viz.add_available_slots(len(T) * SLOT_LENGTH * len(N_0))
    viz.add_obj_val(m.objVal)

    # decision variables

    for n in N_0:
        for t in T:
            for f in F:
                for s in S:
                    if solution_sigma[t, f, n, s] > EPSILON:
                        print 'Optimal sigma\'s for n = %s, t = %s, f = %s, s = %s, equals %s' % (n, t, f, s, solution_sigma[t, f, n, s])
                        viz.add_sigma(t, f, n, s)

    for n in N_0:
        for r in R:
            for mcs in M:
                if solution_beta[r, mcs, n] > EPSILON:
                    print 'Optimal beta\'s for n = %s, r = %s, mcs = %s, equals %s' % (n, r, mcs, solution_beta[r, mcs, n])
                    viz.add_reliability(n, Rel[n, P[n], mcs])
                    viz.add_beta(r, mcs, n)

    for n in N_0:
        if solution_delta[n] > EPSILON:
            viz.add_delta(n, solution_delta[n])
        if solution_arr[n] > EPSILON:
            viz.add_arr(n, solution_arr[n])

    viz.visualize('arrival')

    # CHECKS

    errors = []

    for n in N_0:
        for t in T:
            for f in F:
                for s in S:
                    if solution_sigma[t, f, n, s] > EPSILON:
                        if (t + s - 1) > T_MAX:
                            errors.append('SIGMA[%d, %d, %d, %d], ending at %d, exceeds T_MAX = %d' % (t, f, n, s, (t + s - 1), T_MAX))

    count_sigma = {}

    # check if there is a conflicting sigma
    for n in N_0:
        for t in T:
            for f in F:
                for s in S:
                    if solution_sigma[t, f, n, s] > EPSILON:
                        if n not in count_sigma:
                            count_sigma[n] = 0
                        count_sigma[n] += 1
                        set_of_t = set([t + i for i in range(s)])
                        for _t in T:
                            for _s in S:
                                if _t + (_s - 1) <= T_MAX:
                                    set_of__t = set([_t + i for i in range(_s)])
                                    if len(set_of_t) + len(set_of__t) > len(set_of_t.union(set_of__t)):
                                        # check over n its children
                                        n_and_children = [n]
                                        if n in C:
                                            n_and_children += C[n]
                                        for _n in n_and_children:
                                            # if n != _n:
                                            # check over all frequencies
                                            for _f in F:
                                                # if it is exactly the sigma, you should not throw an error b/c it just the tuple itself that you check
                                                if solution_sigma[_t, _f, _n, _s] > EPSILON and not (_t == t and _f == f and _n == n and _s == s):
                                                    if n in C and _n in C[n]:
                                                        errors.append('Conflicting SIGMA[%d, %d, %d, %d] of child with SIGMA[%d, %d, %d, %d].' % (_t, _f, _n, _s, t, f, n, s))
                                                    else:
                                                        errors.append('Conflicting SIGMA[%d, %d, %d, %d] with SIGMA[%d, %d, %d, %d].' % (_t, _f, _n, _s, t, f, n, s))

        # check if there is a conflicting sigma
        for p in N:
            if p in C:
                for n in C[p]:
                    for t in T:
                        for f in F:
                            for s in S:
                                if solution_sigma[t, f, n, s] > EPSILON:
                                    set_of_t = set([t + i for i in range(s)])
                                    for _t in T:
                                        for _s in S:
                                            if _t + (_s - 1) <= T_MAX:
                                                set_of__t = set([_t + i for i in range(_s)])
                                                # if this is true, you know they do overlap
                                                if len(set_of_t) + len(set_of__t) > len(set_of_t.union(set_of__t)):
                                                    # check over n its children
                                                    siblings = C[p]
                                                    for _n in siblings:
                                                        if _n != n: # do not check with yourself anymore
                                                            # check over all frequencies
                                                            for _f in F:
                                                                if solution_sigma[_t, _f, _n, _s] > EPSILON:
                                                                    errors.append('Conflicting SIGMA[%d, %d, %d, %d] of sibling with SIGMA[%d, %d, %d, %d].' % (_t, _f, _n, _s, t, f, n, s))

    # solution_beta[(1, u'QPSK_FEC_1_2_FR_2', 1)] = 0
    # check if the number of sigma's is <= r
    for n in N_0:
        sigma_count = 0
        for t in T:
            for f in F:
                for s in S:
                    if solution_sigma[t, f, n, s] > EPSILON:
                        sigma_count += 1

        beta_r = None
        beta_mcs = None
        for mcs in M:
            for r in R:
                if solution_beta[r, mcs, n] > EPSILON:
                    if beta_r is not None: # check if there are no multiple r
                        errors.append('Conflicting r of BETA[%d, %s, %d] conflicts with an already assigned r = %d.' % (r, mcs, n, beta_r))
                    else:
                        beta_r = r
                    if beta_mcs is not None: # check if there are no multiple MCSs
                        errors.append('Conflicting MCS of BETA[%d, %s, %d] conflicts with an already assigned MCS %s.' % (r, mcs, n, str(beta_mcs)))
                    else:
                        beta_mcs = mcs

        if not beta_r or not beta_mcs:
            errors.append('There is no BETA for node %d' % (n))
        if beta_r and sigma_count % beta_r != 0:
            errors.append('The SIGMA count (= %d) of node %d is not a multiple of the selected BETA[%d, %s, %d]' % (sigma_count, n, beta_r, beta_mcs, n))
        if beta_r and solution_delta[n] > EPSILON and abs(solution_delta[n] - (sigma_count / float(beta_r))) > EPSILON:
            errors.append('The SIGMA count (= %d) of node %d divided by the selected BETA[%d, %s, %d] does not equal the DELTA (= %d)' % (sigma_count, n, beta_r, beta_mcs, n, solution_delta[n]))

    for n in N_0:
        if solution_delta[n] > EPSILON:
            arr_children = 0.0
            if n in C:
                for child_n in C[n]:
                    if solution_arr[child_n] > EPSILON:
                        arr_children += solution_arr[child_n]
            if solution_delta[n] > (x_n + math.ceil(arr_children)) + EPSILON:
                errors.append('The DELTA (= %d) of node %d is larger than x_n + the arr(children_of_n) = %.4f.' % (solution_delta[n], n, (arr_children + x_n)))

    # equation 9 check
    for n in N_0:
        arr_children = 0.0
        if n in C:
            for child_n in C[n]:
                if solution_arr[child_n] > EPSILON:
                    arr_children += solution_arr[child_n]

        total = x_n + arr_children

        beta_r = None
        beta_mcs = None
        for mcs in M:
            for r in R:
                if solution_beta[r, mcs, n] > EPSILON:
                    beta_r = r
                    beta_mcs = mcs
        if beta_r and beta_mcs: # I am relying on the fact that there should be a beta for each node, this is also tested above.
            total *= fPDR(n, beta_mcs, beta_r, 1)

        # equation 9
        if solution_arr[n] > total + EPSILON:
            errors.append('The ARR of n %d (= %.15f) is larger than the number of packets available times the PDR (= %.15f).' % (n, solution_arr[n], total))
        # equation 10
        transmissions_times_pdr = solution_delta[n] * fPDR(n, beta_mcs, beta_r, 1)
        if solution_arr[n] > transmissions_times_pdr + EPSILON:
            errors.append('The ARR of n %d (= %.4f) is larger than the number of transmissions at node times the PDR (= %.4f).' % (n, solution_arr[n], total))

    # check if the number of packets that arrive at the parent of n is smaller than all the possible x_n
    for n in N_0:
        numberOfTotalChildren = len(getAllChildren(n))
        # print '%d : Total number of children: %d, list %s' % (n, numberOfTotalChildren, getAllChildren(n))
        totalNumberOfPackets = (x_n + (numberOfTotalChildren * x_n))
        if solution_arr[n] > totalNumberOfPackets + EPSILON:
            errors.append('The ARR of n %d (= %.15f) is larger than the number of theoretical packets available (= %.15f).' % (n, solution_arr[n], totalNumberOfPackets))

        packetsThatCanArriveTotal = x_n
        for n_arr in getAllChildren(n):
            if solution_arr[n_arr] > EPSILON:
                packetsThatCanArriveTotal += 1
        if solution_arr[n] > packetsThatCanArriveTotal + EPSILON:
            errors.append('The ARR of n %d (= %.15f) is larger than the number of theoretical packets that could arrive (= %.15f).' % (n, solution_arr[n], packetsThatCanArriveTotal))

    # check if there is no "gap" of sigma's in path of nodes: grandparent has sigmas, parent not, but child has. Should not happen.
    for child, parent in P.iteritems():
        if child in count_sigma:
            # you can not have a sigma if your parent does not have sigma, unless you are the root (who does not have sigma's)
            if count_sigma[child] > 0 and ((parent not in count_sigma) and parent != 0):
                errors.append('The SIGMA count of child %d is %d while that of parent %d is 0.' % (child, count_sigma[child], parent))

    ilp_solution = {}
    ilp_solution['MIPGap'] = m.MIPGap
    ilp_solution['runtime'] = m.runtime
    ilp_solution['setMIPGap'] = m.Params.MIPGap
    ilp_solution['setTimeLimit'] = m.Params.TimeLimit
    ilp_solution['setThreads'] = m.Params.Threads
    ilp_solution['errors'] = errors
    ilp_solution['nrSolutions'] = m.SOLCOUNT
    ilp_solution['objVal'] = m.objVal
    ilp_solution['throughputVal'] = calculated_throughput

    if GRB.Status.TIME_LIMIT == status:
        ilp_solution['hitTimeLimit'] = True
    else:
        ilp_solution['hitTimeLimit'] = False

    with open('ilp_solution.json', 'w') as outfile:
        json.dump(ilp_solution, outfile)

    if len(sys.argv) > 1:
        schedule = dict()
        for n in N_0:
            if n not in schedule:
                schedule[n] = dict()
            n_mcs = None
            for r in R:
                for mcs in M:
                    if solution_beta[r, mcs, n] > EPSILON:
                        n_mcs = mcs  # get MCS for this node
            for t in T:
                for f in F:
                    for s in S:
                        if solution_sigma[t, f, n, s] > EPSILON:
                            if t not in schedule[n]:
                                schedule[n][t] = dict()
                            if f not in schedule[n][t]:
                                schedule[n][t][f] = dict()
                                schedule[n][t][f]['mcs'] = n_mcs
                                schedule[n][t][f]['slots'] = s
                                schedule[n][t][f]['parent'] = P[n]

        schedule_file = str(sys.argv[2])

        milp_schedule = {}
        milp_schedule['schedule'] = schedule
        with open(schedule_file, 'w') as outfile:
            json.dump(milp_schedule, outfile)

# except GurobiError as e:
#     print('Error code ' + str(e.errno) + ": " + str(e))

# except AttributeError as e:
#     print('Encountered an attribute error: ' + str(e))
