import glob, os
video_list = [
    "BasketballDrill_832x480_50.yuv",
    "PartyScene_832x480_50.yuv",
    "Johnny_1280x720_60.yuv"
]
tot_frm = 4

for video_name in video_list:
    if not os.path.exists("./C2/"+video_name):
        os.mkdir("./C2/"+video_name)
    for name in glob.glob("./C2/ori_"+video_name+"/*"):
        input_file = open(name)
        files = []
        for frame in range(tot_frm):
            files.append(open("./C2/"+video_name+"/"+name.split("/")[-1].split(".")[0]+"_"+str(frame)+".txt","w"))
        lines = input_file.readlines()
        for line in lines:
            poc_number = line.split(" ")[0]
            if int(poc_number)==-1:
                break
            files[int(poc_number)].write(line)
        for each_file in files:
            each_file.write(str(-1)+" "+str(-1)+" "+str(-1)+" "+str(-1)+" "+str(-1)+"\n")
          