import sys


print str(sys.argv)
try:
   print int(sys.argv[1])
except:
    print("Make sure your input arg is a number!")

