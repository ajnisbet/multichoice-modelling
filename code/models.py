#!/usr/local/bin/python
# -*- coding: utf-8

import os
import PythonMagick
import re
import string
import subprocess

from cStringIO import StringIO
from pdfminer.converter import  TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from PyPDF2 import PdfFileWriter, PdfFileReader


class Paper(object):
    '''A paper containing multiple questions. Gets built from a paper metadata csv file'''
    def __init__(self, exam_name, subject, year, filepath, first_page, last_page, n_questions, question_regex, n_answers_is_fixed, n_answers, answer_regex, solutions):
        self.exam_name = exam_name
        self.subject = subject
        self.year = year
        self.filepath = filepath
        self.first_page = int(first_page)
        self.last_page = int(last_page)
        self.n_questions = int(n_questions)
        self.question_regex = re.compile(question_regex, re.MULTILINE)
        self.questions = []
        self.n_answers_is_fixed = n_answers_is_fixed=='True'
        self.n_answers = int(n_answers) if n_answers else None
        self.answer_regex = re.compile(answer_regex, re.MULTILINE)
        self.mined_text = ''
        self.ocr_text = ''
        self.solutions = {i+1:a.upper() for i, a in enumerate(list(solutions))}

    def trim_pages(self):
        '''Saves a copy of the paper trimmed to first page and last page, as defined in paper metadata'''


        # Paper is saved with a new filename
        output_filepath = self.filepath.replace('.pdf', '_trimmed.pdf')
        
        self.filepath = output_filepath
        return

        with open(self.filepath, 'rb') as infile, open(output_filepath, 'wb') as outfile:
            # Build readers and writers
            pdf_reader = PdfFileReader(infile)
            pdf_writer = PdfFileWriter()

            # Rewrite, with 0-index fix
            for i in xrange(self.first_page-1, self.last_page):
                pdf_writer.addPage(pdf_reader.getPage(i))

            # Save new pdf
            pdf_writer.write(outfile)
            self.filepath = output_filepath

    def mine(self):
        '''Mines pdf, and adds text as a property'''

        with open(self.filepath, 'rb') as f:
            # PDFminer setup
            parser = PDFParser(f)
            document = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            laparams = LAParams()
            # laparams = LAParams(line_margin=0.3)
            codec = 'utf-8'
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(document):
                    interpreter.process_page(page)
            self.mined_text = retstr.getvalue()
            retstr.close()

    def convert_to_tiff(self):
        return
        '''Converts pdf to BW tiff'''
        input_filepath = self.filepath.replace(' ', '\ ')
        output_filepath = self.filepath.replace('.pdf', '.tif').replace(' ', '\ ')

        command = (
            'gs'
            ' -q'
            ' -r600'
            ' -dNOPAUSE'
            ' -sDEVICE=tiffg4'
            ' -dBATCH'
            ' -sOutputFile={0}'
            ' {1}'
        ).format(output_filepath, input_filepath)

        os.system(command)

    def ocr(self):
        '''Calls tiff converter, then extracts text using tessaract, addind as a property'''
        self.convert_to_tiff() #Remove to help speed
        input_filepath = self.filepath.replace('.pdf', '.tif').replace(' ', '\ ')

        command = (
            'tesseract'
            ' {0}'
            ' stdout'
            ' -psm 6'
            ' /mnt/data/Dropbox/Uni/2014/STAT\ 462/Multichoice\ Project/config/myconfig'
        ).format(input_filepath)

        self.ocr_text = subprocess.check_output(command, shell=True)
        # print self.ocr_text


    def build_questions(self):
        '''
        First builds question from mined text,
        then fills in the other questions with info
        from the ocr text, saving the result in self.questions
        '''

        # First check both sources have been compited
        if not self.mined_text:
            self.mine()
        if not self.ocr_text:
            self.ocr()

        # The get questions for each source
        self.mined_questions = self.get_questions(source='mine')
        self.ocr_questions = self.get_questions(source='ocr')

        # Add questions, first from mined then from ocr
        for i in xrange(1, self.n_questions+1):

            # Try to add mined question
            valid_mined_questions = [q for q in self.mined_questions if q.number == i]
            valid_ocr_questions = [q for q in self.ocr_questions if q.number == i]
            if valid_mined_questions:
                self.questions.append(valid_mined_questions[0])
            elif valid_ocr_questions:
                self.questions.append(valid_ocr_questions[0])


    def get_questions(self, source='mine'):
        '''
        Goes throught the text extracting the questions.
        Also adds answers to those questions.
        Returns a list of Question objects
        '''

        if source == 'ocr':
            text = self.ocr_text
        else:
            text = self.mined_text

        # Find questions using metadata regex, skipping the first in the split
        questions = self.question_regex.split(text)
        numbers = questions[1::2]
        numbers = [''.join(c for c in list(n) if c.isdigit()) for n in numbers]
        questions = questions[2::2]

        source_questions = []

        # Add Question objects for each text
        for number, q_text in zip(numbers, questions):
            # question = Question(q_text, self, int(number))
            # self.questions.append(question)
            try:
                question = Question(q_text, self, int(number), source)
                assert not any(a.text=='' for a in question.answers)
                assert 2 <= len(question.answers) <= 5
                source_questions.append(question)
            except Exception as e:
                print e
                pass


        # Remove duplicate question numebrs
        numbers = [q.number for q in source_questions]
        duplicates = [n for n in numbers if numbers.count(n)>1]
        if duplicates:
            source_questions = [q for q in source_questions if q.number not in duplicates]


        # Now check the ordering
        if len(set(numbers))!=len(numbers) or numbers!=sorted(numbers):
            print 'Questions are not in order'

        return source_questions

            # fixed_questions = [self.questions[0]]
            # for i in xrange(1, len(self.questions)):
            #   if self.questions[i-1].number < self.questions[i].number < self.questions[i+1].number:
            #       fixed_questions.append(self.questions[i])
            # fixed_questions.append(self.questions[-1])
            # self.questions = fixed_questions




class Question(object):
    '''A question, with many answers'''
    def __init__(self, text, paper, number, source):
        self.text = text.strip()
        self.question = paper.answer_regex.split(self.text)[0].strip()
        self.question = ' '.join(self.question.split())
        self.number = number
        self.source = source
        
        self.answers = []
        self.build_answers(paper)
        self.n_answers = len(self.answers)
        self.solution = paper.solutions[number]

        # Computed properties
        self.is_inverse_logic = 'NOT' in self.question
        self.add_relative_answer_properties()

    def __str__(self):
        return 'Question {}: "{}"'.format(self.number, self.question)

    def build_answers(self, paper):
        '''Uses the answer regex to extract the answer text, the builds Answer objects'''
        answers = paper.answer_regex.split(self.text)[2::2]
        answers = [a.rstrip() for a in answers]

        # Make sure the right number of answers is found
        if paper.n_answers_is_fixed:
            assert len(answers) == paper.n_answers, 'The wrong number of answers were found: ({} not {})'.format(len(answers), paper.n_answers) + str(answers) + self.text

        # If there are multiple newlines in the last A (and none in any of the others), split it up
        if not any(['\n\n' in a for a in answers[:-1]]):
            answers[-1] = answers[-1].split('\n\n')[0].strip()
        answers[-1] = answers[-1].split('\x0c')[0].strip()


        # Add answer objects for each text
        for index, a_text in enumerate(answers):
            answer = Answer(a_text, index, paper, self)
            self.answers.append(answer)

    def add_relative_answer_properties(self):
        '''
        Goes through all the answer properties that have some ordinal definition
        and gives the answers indices from 0 to 1, prefixed by 'relative_'
        Also adds 'positional_' that is uniformly distributed
        '''

        def normalise(array):
            old_min = min(array)
            array = [i - old_min for i in array]
            old_max = float(max(array))
            array = [i / old_max for i in array]
            return array

        def add_relative_property(name):
            for a in self.answers:
                # Check names aren't being overwritten
                assert not hasattr(a, 'positional_' + name)
                assert not hasattr(a, 'relative_' + name)

                # Defualt to none
                setattr(a, 'positional_' + name, None)
                setattr(a, 'relative_' + name, None)

            if all([getattr(a, name) is not None for a in self.answers]):
                indices = [i[0] for i in sorted(enumerate(self.answers), key=lambda x:getattr(x[1], name))]
                values = [getattr(i[1], name) for i in sorted(enumerate(self.answers), key=lambda x:getattr(x[1], name))]

                # Meaningless if they're all the same
                if len(set(values)) == 1:
                    return

                indices = normalise(indices)
                values = normalise(values)
                for i, a in enumerate(self.answers):
                    setattr(a, 'positional_' + name, indices[i])
                    setattr(a, 'relative_' + name, values[i])

        relative_properties = ['value', 'n_words', 'n_long_words', 'length' ]
        for name in relative_properties:
            add_relative_property(name)

        # Index properties
        for a in self.answers:
            a.is_first = False
            a.is_last = False
        self.answers[0].is_first = True
        self.answers[-1].is_last = True





class Answer(object):
    '''A paper answer'''
    def __init__(self, text, index, paper, question):
        self.text = ' '.join(text.split())
        self.index = index # Ordinal position in list of answers
        self.letter = string.ascii_uppercase[self.index] #upperace letter based on index in list of ansers
        self.source = question.source

        # computed properties
        self.is_correct = self.letter == paper.solutions[question.number]
        self.add_numeric_properties()
        self.add_length_properties()
        self.add_logic_properties()


    def __str__(self):
        return '\t{}: "{}" [{}, {}] {}'.format(self.letter, self.text, self.value, self.relative_value, '✔' if self.is_correct==True else '✖')

    def add_numeric_properties(self):
        '''Figures out if the answer text represents a numerical value'''
        
        self.value = None
        self.is_numeric = False
        self.is_integer = False
        self.is_float = False
        self.is_fraction = False
        self.is_percent = False
        self.is_unit = False

        text = self.text.rstrip('.')

        try:
            self.value = int(text)
            self.is_numeric = True
            self.is_integer = True
            return
        except:
            pass

        try:
            self.value = float(text)
            self.is_numeric = True
            self.is_float = True
            return
        except:
            pass

        if text.replace('/', '', 1).strip().isdigit():
            parts = text.split('/')
            self.value = float(parts[0]) / float(parts[1])
            self.is_numeric = True
            self.is_fraction = True
            return

        if text.replace('%', '', 1).strip().isdigit():
            self.value = float(text.replace('%', '').strip())
            self.is_numeric = True
            self.is_percent = True 
            return

        # Check all answers have the same units?
        units = [
            'minute',
            'minutes',
            'mm',
            'mL',
            'kJ',
            'eV',
            'g',
            'millimetres',
            'micrometres',
            'cm 3 per minute',
            'o C',
            'oC',
            'cm3 per minute',
            'mol kg–1',
            'm s−2',
        ]
        
        for unit in units:
            no_unit_text = text.replace(unit, '').strip()

            try:
                self.value = float(no_unit_text)
                self.is_numeric = True
                self.is_unit = True
                return
            except:
                pass



    def add_length_properties(self):

        # Basic length
        self.length = len(self.text)

        # Extract words
        words = self.text.split()

        # Remove non-words from words list
        exclude = '.,?-\'"()+-/!@#$%^&*()_1234567890'
        unpunctuate = lambda w: ''.join(c for c in w if c not in exclude)
        words = [unpunctuate(w) for w in words]
        words = [w for w in words if w]
        self.n_words = len(words)

        # Look at long words
        long_word_min_length = 4
        long_words = [w for w in words if len(w) >= long_word_min_length]
        self.n_long_words = len(long_words)

    def add_logic_properties(self):

        # Check if an answer says none of the answers are good
        none_text = [
            'there would be no ',
            'none of the',
            'none,'
            'it is impossible to tell'
        ]
        self.is_none_of_the_above = any([self.text.lower().startswith(t) for t in none_text])


