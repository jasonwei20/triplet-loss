
def clean_categories_file(file_path, output_path):
	lines = open(file_path, 'r').readlines()
	
	q_to_category = {}
	categories = set()
	for line in lines:
		parts = line[:-1].split(',')
		q = parts[0]
		category = parts[1]
		q_to_category[q] = category
		categories.add(category)

	categories = list(sorted(categories))
	category_to_id = {categories[i]: i for i in range(len(categories))}

	output_writer = open(output_path, 'w')

	for q, category in q_to_category.items():

		_id = category_to_id[category]
		output_writer.write('\t'.join([str(_id), q]) + '\n')

def clean_clusters_file(file_path, output_path, category_to_id_resume):
	lines = open(file_path, 'r').readlines()
	
	q_to_category = {}
	clusters_counter = {}
	for line in lines:
		parts = line[:-1].split(',')
		cluster = parts[0]
		q = parts[1]
		q_to_category[q] = cluster
		if cluster in clusters_counter:
			clusters_counter[cluster] += 1
		else:
			clusters_counter[cluster] = 1

	clusters_counter = {k: v for k, v in clusters_counter.items() if v == 3}
	categories = list(sorted(clusters_counter.keys()))
	category_to_id = {categories[i]: i for i in range(len(categories))}

	if category_to_id_resume:
		category_to_id = category_to_id_resume

	output_writer = open(output_path, 'w')

	for q, cluster in q_to_category.items():

		if cluster in category_to_id:
			_id = category_to_id[cluster]
			output_writer.write('\t'.join([str(_id), q]) + '\n')

	print(category_to_id)
	return category_to_id

if __name__ == "__main__":
	# clean_categories_file("categories_train.csv", "covid_cat/train.txt")
	# clean_categories_file("categories_test.csv", "covid_cat/test.txt")
	category_to_id = clean_clusters_file("class_train.csv", "covid_clus/train.txt", None)
	clean_clusters_file("class_test.csv", "covid_clus/test.txt", category_to_id)
