import csv
import os
import pickle
import operator
import numpy as np
import subprocess
import matplotlib.pyplot as plt


from models import Paper

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.utilities import fListToString



def make_papers():
	'''Create and return Paper objects for all papers in the metadata file'''

	# Name of metadata file
	metadata_filename = 'paper_metadata.csv'
	papers = []

	# Read file
	with open(metadata_filename) as f:
		lines = [line.rstrip('\n') for line in f]
	
	# Build papers, skipping the first line
	for line in lines[1:]:
		row = line.split(',')
		row[3] = os.path.join(os.getcwd(), row[3])
		papers.append(Paper(*row))

	return papers

def list_authors():
	'''Lists all the authors of the PDFs'''
	# Name of metadata file
	metadata_filename = 'paper_metadata.csv'
	authors = []

	# Read file
	with open(metadata_filename) as f:
		lines = [line.rstrip('\n') for line in f]
	
	# Build papers, skipping the first line
	for line in lines[1:]:
		row = line.split(',')
		filepath = os.path.join(os.getcwd(), row[3])
		output = subprocess.Popen(['pdfinfo', filepath], stdout=subprocess.PIPE).communicate()[0]
		try:
			authors.append([l.strip() for l in output.split('\n') if l.startswith('Author')][0])
		except:
			print output

	for a in authors:
		print a





def create_data():

	# Make paper objects of everything in the metadata file
	papers = make_papers()
	papers = papers[1:]
	
	# Iterate over papers
	for paper in papers:

		# Remove the wrong papers
		paper.trim_pages()

		# Make questions from mined text
		paper.build_questions()

		# Pretty printing 
		# for question in paper.questions:
		# 	print question, '('+question.source+')'
		# 	for answer in question.answers:
		# 		print answer
		# 	print '\n'

		print '{} / {} questions found'.format(len(paper.questions), paper.n_questions)
		print [q.number for q in paper.questions]
		print [q.number for q in paper.mined_questions]
		print [q.number for q in paper.ocr_questions]

	# Now save the papers
	pickle.dump(papers, open('papers2.p', 'wb'))



def display_data():
	papers = pickle.load(open('papers2.p'))

	actual_n = 0
	n_found = 0
	n_mined = 0
	n_ocr = 0

	subject_totals = {'Biology': 0, 'Chemistry': 0, 'Physics': 0}

	for paper in papers:
		actual_n += paper.n_questions
		n_found += len(paper.questions)
		n_mined += len(paper.mined_questions)
		assert len(paper.questions) >= len(paper.mined_questions)
		n_ocr += len(paper.questions) - len(paper.mined_questions)
		subject_totals[paper.subject] += len(paper.questions)

		print ' '.join([paper.subject, paper.year])
		print '{} / {} questions found'.format(len(paper.questions), paper.n_questions)
		print [q.number for q in paper.questions]
		print [q.number for q in paper.mined_questions]
		print [q.number for q in paper.ocr_questions]
		print '\n'

	print 'Questions in dataset: {}'.format(actual_n)
	print 'Questions found:      {}'.format(n_found)
	print 'Mined questions:      {}'.format(n_mined)
	print 'OCR questions:        {}'.format(n_ocr)
	print subject_totals


def process_data():
	papers = pickle.load(open('papers2.p'))

	X = []
	Y = []

	for paper in papers:
		for question in paper.questions:
			for answer in question.answers:

				# Get the list of answer paremeters, and sort so they are the same order for all anwsers
				sorted_vars = sorted(vars(answer).iteritems(), key=operator.itemgetter(0))

				# Remove anything with a string
				sorted_vars = [(key, value) for (key, value) in sorted_vars if type(value) != str]

				# Remove the answer :)
				sorted_vars = [(key, value) for (key, value) in sorted_vars if key != 'is_correct']

				# Coerce booleans to integers
				# sorted_vars = [(key, int(value)) if type(value)==bool else (key, value) for (key, value) in sorted_vars]


				# Extract the values
				just_values = [value for (key, value) in sorted_vars]
				X.append(just_values)

				# And append the answer
				Y.append(int(answer.is_correct))

	# Normalise results
	X = np.array(X)
	n_rows, n_cols = X.shape

	def normalise(value, maxx, minn):
		if type(value) == bool:
			return value
		rangee = float(maxx - minn)
		if rangee == 0:
			return 0 * value
		return ((value - minn) / rangee - 0.5 )* 2




	for i in xrange(n_cols):
		column = X[:, i]
		col_max = max([c for c in column if c is not None])
		col_min = min([c for c in column if c is not None])
		X[:, i] = [normalise(c, col_max, col_min) if c is not None else 0 for c in column]
		print min(X[:, i]), max(X[:, i])

	# Figue out expected result
	total = 0
	n_questions = 0
	for paper in papers:
		for question in paper.questions:
			n_questions += 1
			total += 1. / len(question.answers)

	expected = total / n_questions

	return X, Y

def do_nn(X, Y):

	# Number of epochs
	n_epochs = 10

	# Move to numpy array
	X = np.array(X)
	Y = np.array(Y)
	n_rows, n_cols = X.shape
	n_observations, n_features = n_rows, n_cols

	# Build dataset
	alldata = ClassificationDataSet(n_features, nb_classes=2)

	# Add data to dataset
	for i in xrange(n_observations):
		alldata.addSample(X[i, :], Y[i])

	# Split into training and testing
	testing_data, training_data = alldata.splitWithProportion(0.25)

	# Encode classes with one output neuron per class
	training_data._convertToOneOfMany()
	testing_data._convertToOneOfMany()

	# Built the buildNetwork
	fnn = buildNetwork(training_data.indim, training_data.outdim, outclass=SoftmaxLayer)
	print fnn

	n = FeedForwardNetwork()

	inLayer = LinearLayer(training_data.indim)
	hiddenLayer = LinearLayer(training_data.indim/2)
	outLayer = SigmoidLayer(training_data.outdim)

	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outLayer)

	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outLayer)
	in_to_out = FullConnection(inLayer, outLayer)

	n.addConnection(in_to_hidden)
	n.addConnection(hidden_to_out)
	n.addConnection(in_to_out)

	n.sortModules()

	# Set up a trainer
	trainer = BackpropTrainer(fnn, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

	# Start training
	train_errors, validation_errors = trainer.trainUntilConvergence(maxEpochs=100)
	train_errors = trainer.trainingErrors
	validation_errors = trainer.validationErrors[:-1]
	plt.figure()
	plt.plot(xrange(len(train_errors)), train_errors)
	plt.plot(xrange(len(validation_errors)), validation_errors)
	plt.title('Neural Netwok Error')
	plt.xlabel('Number of iterations')
	plt.ylabel('RMSE')
	plt.legend(['Testing', 'Validation'])
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,x2,0.06,y2))

	plt.show()
	print train_errors
	print validation_errors



	# for i in xrange(n_epochs):
	# 	trainer.trainEpochs(1)
	# 	print trainer.testOnData()
	# 	train_result = percentError(trainer.testOnClassData(), training_data['class'])
	# 	test_result = percentError(trainer.testOnClassData(), testing_data['class'])

	# Now do the whole thing to extract the prediction
	is_correct_weights = fnn.activateOnDataset(alldata)[:, 1]
	return is_correct_weights


def get_results(is_correct_weights):
	papers = pickle.load(open('papers2.p'))

	# Save the wieghts
	i = 0
	for paper in papers:
		for question in paper.questions:
			for answer in question.answers:
				answer.is_correct_weight = is_correct_weights[i]
				i += 1

	# Use the weights to predict the actual value
	for paper in papers:
		for question in paper.questions:
			max_weight = max([a.is_correct_weight for a in question.answers])
			for answer in question.answers:
				answer.is_correct_nn = answer.is_correct_weight == max_weight


	# Figue out expected result
	total_expected = 0
	total_nn = 0
	n_questions = 0.
	for paper in papers:
		for question in paper.questions:
			n_questions += 1
			total_expected += 1. / len(question.answers)
			total_nn += any([a.is_correct_nn==True and a.is_correct for a in question.answers])

	expected_result = total_expected / n_questions
	nn_result = total_nn / n_questions
	print expected_result
	print nn_result


if __name__ == '__main__':
	# list_authors()
	# create_data()
	# display_data()
	X, Y = process_data()
	print X.shape
	is_correct_weights = do_nn(X, Y)
	get_results(is_correct_weights)