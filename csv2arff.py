"""
This script converts csv datasets to ARFF format.
In the ARFF format, one needs to know whether features are numeric or categorical - this is done with a simple heuristic, and thus is not always completely accurate.
No support for Data attributes yet.
Categorical attributes should be encoded as numbers

Author: Pieter Gijsbers
"""
import sys
import math
import numpy
import pandas as pd
import numbers
import arff

NUM_UNIQUE_CATEGORICAL = 10

input_file = sys.argv[1]
output_file = sys.argv[2]
sep = ','

if len(sys.argv)>3:
	sep = sys.argv[3]

if '/' in input_file:
	end_of_folder_path = input_file.rfind('/') + 1
else:
	end_of_folder_path = 0

dataset_name = input_file[end_of_folder_path:-4]
dataset = pd.read_csv(input_file, sep=sep)

column_info = []
integer_columns = []
for column in dataset.columns:
	column_values = dataset[column]
	column_is_integer = (all([isinstance(el, numbers.Number) for el in column_values]) and
			     all([math.isnan(el) or float(el).is_integer() for el in column_values]))

	integer_columns.append(column_is_integer)
	if column_is_integer:
		not_nan_values = column_values[~numpy.isnan(column_values)]
		unique_values = numpy.unique(not_nan_values)
		is_categorical = (len(unique_values) <= NUM_UNIQUE_CATEGORICAL and column_is_integer)
		if is_categorical:
			data_type_str = [str(int(val)) for val in unique_values]
		else:
			data_type_str = 'INTEGER'
	elif all([isinstance(val, str) for val in column_values]):
		if len(set(column_values)) <= NUM_UNIQUE_CATEGORICAL:
			data_type_str = list(set(column_values))
		else:
			data_type_str = 'STRING'
	else:
		data_type_str = 'NUMERIC'

	column_info.append((column, data_type_str))

print('Datatypes have been inferred by the values found in the CSV-file.')
col_types = [col_type if not isinstance(col_type,list) else 'CATEGORICAL' for _,col_type in column_info]
col_type_count = numpy.unique(col_types, return_counts=True) 
print('In total we found',', '.join([str(count)+' '+col_type for (col_type, count) in zip(*col_type_count)]))

column_input = None
while column_input != 's': 
	column_input = None
	print('The following attribute types are inferred for the dataset \'{}\':\n'.format(dataset_name))

	for i, (column, data_type) in enumerate(column_info):
		print('[{:3d}] {}: {}'.format(i, column, data_type))
		unique_values = set(dataset[column])
		random_column_values = numpy.random.choice(list(unique_values), size= min(len(unique_values), 10))
		print(min(len(unique_values), 10),'random unique column values (out of {}):'.format(len(unique_values)), random_column_values)

	column_numbers = [str(i) for i in range(len(column_info))]
	while column_input not in  [*column_numbers, 's']:
		column_input = input('\n To change the attribute type of a column, first insert its number.\n'
	      	      		   'To save the ARFF file, insert \'s\'\n')

	if column_input in column_numbers:
		datatype_input = None
		non_category_input = ['i','integer','r','real','n','numeric','s','string']
		category_input = ['c', 'categorical']
		abort_input = ['a','abort']

		while datatype_input not in [*non_category_input, *category_input,*abort_input]:
			print('\n What datatype should column {} with name "{}" be?'.format(column_input, dataset.columns[int(column_input)]))
			datatype_input = input('Valid options are (not case sensitive): [I]NTEGER, [R]EAL, [N]UMERIC, [S]TRING, [C]ATEGORICAL, [A]BORT\n')

		if datatype_input.lower() in category_input:
			column_name = dataset.columns[int(column_input)]
			column_values = dataset[column_name]
			unique_values = set(column_values)
	
			contains_nan = any([math.isnan(el) for el in unique_values if isinstance(el, float)])
			contains_inf = any([math.isinf(el) for el in unique_values if isinstance(el, float)])
			unique_values = [el for el in unique_values if not isinstance(el, float) or not math.isnan(el) or not math.isinf(el)]
			if contains_nan or contains_inf:
				if contains_nan:
					print('Warning! This column contains NaN values.')
					unique_values.append(float("nan"))
				if contains_inf:
					print('Warning! This column contains Inf values.')
					unique_values.append(float("inf"))
			
			column_info[int(column_input)] = (column_info[int(column_input)][0], [str(val) for val in unique_values])
			print('Column {} with name "{}" type changed to categorical with values {}.'.format(
			       column_input, dataset.columns[int(column_input)],[str(val) for val in unique_values]))
			
		elif datatype_input.lower() not in abort_input:
			if len(datatype_input) == 1:
				datatype_input = [datatype for datatype in non_category_input
						  if datatype.startswith(datatype_input) and len(datatype)>1][0]
			column_info[int(column_input)] = (column_info[int(column_input)][0], datatype_input)
			print('Column {} with name "{}" type changed to {}.'.format(column_input, dataset.columns[int(column_input)],datatype_input))
		else:
			print('Not assigning new datatype to column.')

print('Saving ARFF to file...')
arff_dict = {
	'description':  'This ARFF is automatically generated.\n'
			'The data types of attributes are inferred from the data.\n'
		        'In particular this means there is some uncertainty as to whether '
			'an attribute truly is categorical or numerical.\n',
	'relation': dataset_name,
	'attributes': column_info,
	'data': [row for _, row in dataset.iterrows()]
}

with open('{}.arff'.format(dataset_name),'w') as arff_file:
	arff.dump(arff_dict, arff_file)
