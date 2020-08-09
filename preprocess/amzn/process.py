import pandas as pd
from random import shuffle
import random
random.seed(1)

import re
def get_only_chars(line):

	clean_line = ""

	line = line.replace("’", "")
	line = line.replace("'", "")
	line = line.replace("-", " ") #replace hyphens with spaces
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")
	line = line.lower()

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			clean_line += char
		else:
			clean_line += ' '

	clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
	if clean_line[0] == ' ':
		clean_line = clean_line[1:]

	clean_line = ' '.join(clean_line.split(' ')[:100])

	return clean_line

def clean_items(l):
	return [get_only_chars(x) for x in l]

def process_train_val(train_val_csv_path):

	df = pd.read_csv(train_val_csv_path)
	titles = df['Title'].tolist()
	texts = df['Text'].tolist()
	labels = df['Cat3'].tolist()

	label_to_titles = {}
	title_to_text = {}

	for title, text, label in zip(titles, texts, labels):

		if type(title) == str and type(text) == str and type(label) == str and len(title) > 3 and len(text) > 3 and len(label) > 3:

			title = get_only_chars(title)
			text = get_only_chars(text)
			label = get_only_chars(label)

			title_to_text[title] = text

			if label in label_to_titles:
				label_to_titles[label].add(title)
			else:
				label_to_titles[label] = {title}

	del label_to_titles['unknown']

	label_to_titles = {k:v for k, v in label_to_titles.items() if len(v) >= 6}
	label_to_id = {label: i for i, label in enumerate(list(sorted(label_to_titles.keys())))}

	print(len(label_to_titles))

	train_writer = open('train.txt', 'w')
	test_writer = open('test.txt', 'w')

	for label, titles in label_to_titles.items():

		titles_list = list(titles)
		shuffle(titles_list)

		for title in titles_list[:5]:
			text = title_to_text[title]
			combined_title_text = ' '.join([title, text, title])
			train_writer.write(str(label_to_id[label]) + '\t' + combined_title_text + '\n')

		for title in titles_list[5:10]:
			text = title_to_text[title]
			combined_title_text = ' '.join([title, text, title])
			test_writer.write(str(label_to_id[label]) + '\t' + combined_title_text + '\n')


if __name__ == '__main__':
	process_train_val('train_val_50k.csv')