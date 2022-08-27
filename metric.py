import pickle
import matplotlib.pyplot as plt

def create_histogram(arr,name_fig):
    keys = set(arr)
    dict_hist = dict.fromkeys(keys,0)
    for item in arr:
        dict_hist[item] += 1
    plt.figure(1)
    plt.bar(list(dict_hist.keys()),list(dict_hist.values()))
    plt.savefig(name_fig)
    plt.close()
        
# with open("kq_pred_B.pkl","rb") as f:
#     kq = pickle.load(f)
# with open("kq_pred_C.pkl","rb") as f:
#     kq1 = pickle.load(f)
with open("kq_pred_D.pkl","rb") as f:
    kq2 = pickle.load(f)
# kq.extend(kq1)
# kq.extend(kq2)
sev0 = []
sev1 = []
sev2 = []
sev3 = []

for sample in kq2:
    GT = sample["GT"]
    pred = sample["Pred"]
    
    for index,region in enumerate(GT):
        if region == 1:
            sev1.append(int(pred[index]))
        elif region == 2:
            sev2.append(int(pred[index]))
        elif region == 3:
            sev3.append(int(pred[index]))
        else:
            sev0.append(int(pred[index]))

create_histogram(sev1,"histogram_severity/severity_1.jpg")
create_histogram(sev2,"histogram_severity/severity_2.jpg")
create_histogram(sev3,"histogram_severity/severity_3.jpg")
create_histogram(sev0,"histogram_severity/severity_0.jpg")

count0= 0
count1 = 0
count2 = 0
count3 = 0

for item in sev1:
    if item >= 6 and item <18:
        count1 +=1 
for item in sev0:
    if item < 6:
        count0 += 1
for item in sev2:
    if item >= 18 and  item <40:
        count2 += 1    
for item in sev3:
    if item >= 40:
        count3 +=1
print("Severity 0 --- GT: {} ---True: {} --Accuracy: {}".format(len(sev0),count0,count0/len(sev0)))
print("Severity 1 --- GT: {} ---True: {} --Accuracy: {}".format(len(sev1),count1,count1/len(sev1)))
print("Severity 2 --- GT: {} ---True: {} --Accuracy: {}".format(len(sev2),count2,count2/len(sev2)))
print("Severity 3 --- GT: {} ---True: {} --Accuracy: {}".format(len(sev3),count3,count3/len(sev3)))
