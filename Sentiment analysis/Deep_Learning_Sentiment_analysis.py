import sys
import re
import itertools
import operator

file_in_train=sys.argv[1]
train_file=open(file_in_train,"rt")

file_in_test=sys.argv[2]
test_file=open(file_in_test,"rt")

#Read and return Training file 
def return_training_samples(train_file):
	infile=train_file
	tot_lst=[]
	while True:
		line_trxn=infile.readline()
		line_trxn=line_trxn.strip('\n')
		items=(line_trxn.split("	"))
		if not line_trxn:
			break
		tot_lst.append(items)
	return sorted(tot_lst)

def return_senti(list_review):
	dict_senti={}
	sentiment=[]
	for l in list_review:
		if l[-1] not in sentiment:
			sentiment.append(l[-1])
	for l1 in sentiment:
		text_value=[]	
		for l in list_review:
			if l1==l[-1]:
				l[-2]=l[-2].replace(",","")
				l[-2]=re.sub(' +',' ',l[-2])
				text_value.append(l[-2].strip())
		dict_senti[l1]=text_value
	return dict_senti


def return_test_file(test_file):
	infile=test_file
	tot_lst=[]
	while True:
		line_trxn=infile.readline()
		line_trxn=line_trxn.strip('\n')
		items=(line_trxn.split("	"))
		if not line_trxn:
			break
		tot_lst.append(items)
	return sorted(tot_lst)


def return_test_samples(test_file_lst):
	infile=test_file
	tot_lst=[]
	for line in test_file_lst:
		line[-2]=line[-2]+'\n'
		tot_lst.append(line[-2]) #change to tot_lst.append(line[-1]) if test data doesnt have class at the last column  
	dict_sentiment_count={}
	max_sentiment=[]
	count=0
	return(tot_lst)


def give_context_combination(words):
		combo1=''
		c=[]
		for word in words:
					combo1 += word+" "
					c=c+[combo1]
		return(c)

def getsenti_test_samples(tot_lst,train_dict):
	dict_sentiment_count={}
	max_sentiment=[]
	count=0
	for l in tot_lst:
			test_sentence_brk_lst=(re.split(r'[.!?]+', l))
			for s in test_sentence_brk_lst:
				all_permutations=[]
				re_pattern = r'\b\w+\b'
				words = re.findall(re_pattern, s)
				a=[]
				l=''
				m=[]
				for i in range(len(words)):
					a.append(give_context_combination(words))
					for i in a:
						m=m+i
					m.append(words[0])
					if i==0:
						l=words[0]
					elif	i==(len(words)-1):
						l=l
					else:
						l=l+' '+words[0]
					words.remove(words[0])
					m.append(l)
				all_permutations=list(set(m))
				all_p=[]
				for ap in all_permutations:
					all_p.append(ap.strip())
				for k,v in train_dict.items():
					match=(set(all_p).intersection(v))
					count=0
					c=0
					for m in match:
							c+=len(m.split())
							if len(m)==1:
								count+=1
					if k!=2:
							dict_sentiment_count[k]=c+count
			m=sorted(dict_sentiment_count.items(),key=lambda x: (-x[1], x[0]))[0][0]
			if dict_sentiment_count.get(m)==0:
				max_sentiment.append('2')
			else:
				max_sentiment.append(m)
			dict_sentiment_count={}
	return(max_sentiment)


def calc_stats(predicted_sentiment,test_file_lst):
	wrong_classified=0
	count=0
	for i in range(len(predicted_sentiment)):
		count+=1
		if predicted_sentiment[i]!=test_file_lst[i][-1]:
			wrong_classified+=1
	accuracy_total=(1-(wrong_classified/len(predicted_sentiment)))*100

	print('\nTotal Sample: ',len(predicted_sentiment))
	print('Correctly classified: ',(len(predicted_sentiment)-wrong_classified))
	print('Wrong Classified: ',wrong_classified)
	print('Accuracy with matching training data: ',round(accuracy_total,2))
	
	return

def create_conf_matrix(predicted_sentiment, test_file_lst, train_dict):
	expected=[]
	predicted=[]
	n_classes=len(train_dict.keys())
	for i in range(len(test_file_lst)):
		expected.append(int(test_file_lst[i][-1]))
	#print(expected)
	for i in range(len(test_file_lst)):
		predicted.append(int(predicted_sentiment[i]))
	m = [[0] * n_classes for i in range(n_classes)]
	for pred, exp in zip(predicted, expected):
		m[pred][exp] += 1
	return m

def main():
	list_review=return_training_samples(train_file)
	train_dict=return_senti(list_review)
	test_file_lst=return_test_file(test_file)
	tot_lst=return_test_samples(test_file_lst)
	predicted_sentiment=getsenti_test_samples(tot_lst,train_dict)
	accuracy=calc_stats(predicted_sentiment,test_file_lst)
	conf_matrix=create_conf_matrix(predicted_sentiment, test_file_lst, train_dict)
	print('Confusion Matrix: ')
	for c in conf_matrix:
		print(c)
		
	
if __name__ == '__main__':
    main()

