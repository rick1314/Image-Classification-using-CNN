import os, sys
import tensorflow as tf
import json
import pprint
import csv
from fnmatch import fnmatch
import time
from multiprocessing import Process, Lock, Value

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
def findingCat(image_path,dfile):
    print('hello ', image_path,' the ',dfile)
    sys.stdout.flush()
"""

def findcat(name):
    with open('../val2017.json', 'r') as data:
        json_data = json.load(data)
        for img in json_data['categories']:
            if(img['name'].lower().replace('-',' ') == name.lower()):
                #print(img['name'], ' - ',img['id'])
                return img['id']
        data.flush()
        data.close()


def findimgID(name):
    with open('../val2017.json', 'r') as data:
        json_data = json.load(data)
        for img in json_data['images']:
            if(img['file_name'].lower().endswith(name.lower())):
                #print(img['id'])
                return img['id']
        data.flush()
        data.close()

def findorgID(imgid):
    with open('../val2017.json', 'r') as data:
        json_data = json.load(data)
        for img in json_data['annotations']:
            if(img['image_id'] == imgid):
                #print(img['id'])
                return img['category_id']
        data.flush()
        data.close()


def csvwrite(res,csvfile):
    with open(csvfile, "a", newline='') as output:
        writer = csv.writer(output, delimiter=",")
        writer.writerow(res)
        output.flush()
        output.close()


def findingCat(l,image_path,dfile,top1,top3,top5):
	

	l.acquire()
    #image_path = sys.argv[1]
	linedata = []
    #print(findimgID(os.path.basename(image_path)))
    #linedata.append(os.path.basename(image_path))
	imgid = findimgID(os.path.basename(image_path))
	orID = findorgID(imgid)
	#print("Original ID- ",orID)
	linedata.append(imgid)
    # Read in the image_data
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("../retrained_labels.txt")]

    # Unpersists graph from file
	with tf.gfile.FastGFile("../retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
		f.flush()
		f.close()

	with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
		predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		
		count = 0
		for node_id in top_k:
			count+=1
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
            #print('%s (score = %.5f)' % (human_string, score))
			imgcat = findcat(human_string)
            #linedata.append(human_string)
			linedata.append(imgcat)
			if imgcat == orID:
				if count == 1:
					top1.value += 1
				elif count <= 3:
					top3.value += 1
				elif count <= 5:
					top5.value += 1
			if count == 5:
				break
		sess.close()

	print(linedata)
	csvwrite(linedata,dfile)
	l.release()
	sys.stdout.flush()


if __name__ == "__main__":

	dir_path = os.getcwd()
	dirs = os.listdir(dir_path)
	dfile = 'result.csv'
	# This would print all the files and directories
	#for files in dirs:
	#    if(files.lower().endswith(('.jpg', '.jpeg'))):
	#        #print(dir_path+'\\'+files)
	#        findingCat(dir_path+'\\'+files,dfile)
	top1 = Value('i',0)
	top3 = Value('i',0)
	top5 = Value('i',0)
	numim = 0
	lock = Lock()
	for path, subdirs, files in os.walk(dir_path):
		for name in files:
			if name.endswith(('.jpg', '.jpeg')):
				numim += 1
                #findingCat(path + '\\' + name, dfile)
				p = Process(target=findingCat, args=(lock, path + '\\' + name, dfile, top1, top3, top5))
				p.start()
				p.join()
				#if numim == 15:
				#	time.sleep(3) 
	            #print(path+'\\'+name)
	print("=====Performance=====")
	print("Top 1- ",top1.value*100.0/numim)
	print("Top 3- ",top3.value*100.0/numim)
	print("Top 5- ",top5.value*100.0/numim)
	print("Total- ",numim)
"""
if __name__ == '__main__':
    p = Process(target=findingCat, args=('bob','builder'))
    p.start()
    p.join()
"""