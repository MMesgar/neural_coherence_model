import sys
import numpy as np 
def drawProgressBar(shell_out, 
                    begin, k, out_of, end, barLen =25):
    percent = k/float(out_of)
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        elif i==int(barLen * percent):
            progress +='>'
        else:
            progress += "_"
    text = "%s%d/%d[%s](%.2f%%)%s"%(begin,k,out_of,progress,percent * 100, end)
    if shell_out== True:
        sys.stdout.write(text)
        sys.stdout.flush()
    return text