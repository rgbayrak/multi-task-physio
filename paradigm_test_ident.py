# Compare training and test scans so there is no overlap

# current_scan = 'boo'
# task_subject = []
# f = open("/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/rest.txt", "r")
# rest = f.readlines()
# task_subject.extend([x.strip() for x in rest])
#
# lst = []
# f = open("/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/wm.txt", "r")
# lst = f.readlines()
#
# h = open("/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/o_wm.txt", "a")
# for scan in lst:
#     for sub in task_subject:
#         if sub in scan:
#             if scan != current_scan:
#                 current_scan = scan
#                 h.write(scan)
# h.close()

with open('/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/language.txt', 'r') as file1:
    with open('/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/o_language.txt', 'r') as file2:
        not_same = set(file1).symmetric_difference(file2)

no = sorted(not_same)
with open("/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans/no_language.txt", "w") as output:
    for row in no:
        output.write('rois_{}.mat\n'.format(row.strip()))
pass