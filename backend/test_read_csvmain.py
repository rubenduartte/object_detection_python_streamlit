import csv
import time
import os


class FileTailer(object):
    def __init__(self, file, delay=0.1):
        self.file = file
        self.delay = delay
    def __iter__(self):
        while True:
            where = self.file.tell()
            line = self.file.readline()
            if line and line.endswith('\n'): # only emit full lines
                yield line
            else:                            # for a partial line, pause and back up
                time.sleep(self.delay)       # ...not actually a recommended approach.
                self.file.seek(where)

file_dir = os.path.dirname(os.path.realpath('__file__'))
file_name = os.path.join(file_dir,  r'runs\detect\train\results.csv')


csv_reader = csv.reader(FileTailer(open(file_name)))
for row in csv_reader:
    print("Read row: %r" % (row,))




# # def csv_cycle(file):
# #     with open(file) as fd:
# #         reader = csv.DictReader(fd)
# #         while True:
# #             for row in reader:
# #                 yield row
# #             fd.seek(0)
# #             fd.readline() # skips the header

# def tail(f):
#     f.seek(0, 2)

#     while True:
#         line = f.readline()

#         if not line:
#             time.sleep(0.1)
#             continue

#         yield line

 
# file = "results.csv"
# with open(file) as fd:
#     print(fd)

# while True:
#     auditlog = tail(open(file) )
#     for line in auditlog:
#         print(line)
 
# # for row in csv_cycle(file):
# #     # do something with row
# #     time.sleep(0.5)
# #     print(row)