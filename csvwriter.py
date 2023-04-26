import glob
import csv

files=glob.glob('*/*/*')
files_name = []
counter = 0
label = 0

for i in files:
    files_name.append(i)

files_name.remove(files_name[-1])

with open('products.csv', mode='w',newline='') as yeni_dosya:
    writer_names = csv.writer(yeni_dosya, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer_names.writerow(["asd","4"])
    for i in files_name:
        counter += 1
        writer_names.writerow([i,label])
        if counter == 50:
            counter = 0
            label += 1







