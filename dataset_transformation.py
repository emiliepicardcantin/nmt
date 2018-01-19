import json
import pickle
from tqdm import tqdm
import re
import csv
import time

import scraping_with_stanford


output_path = '/tmp/nmt_data/content_to_headlines/'
vocab_size = 10000

def get_vocab(lst):
    vocabcount = {}
    vocab = []
    for ws in tqdm(lst):
        line = re.sub(r'[|.,:;"?!@#%%^&\*()_\-+={}\[\]/<>\n]', str(' '), ws)
        for w in line.split():
            vocabcount[w] = vocabcount.get(w,0)+1 
    vocab = [x[0] for x in sorted(vocabcount.items(), key=lambda x: -x[1])]
    return vocab, vocabcount

def create_vocabularies(folder,extension):
    print("Getting "+extension+" vocabulary")

    words = []
    with open(folder+'train.'+extension, 'r') as train_file:
        train_lines = [line for line in train_file]
    with open(folder+'test.'+extension,'r') as test_file:
        test_lines = [line for line in test_file]

    words = train_lines + test_lines

    print("Loaded all words")

    vocab,vocab_count = get_vocab(words)

    with open(folder+'vocab.'+extension,'w') as vocab_file:
        vocab_file.write('<unk>\n<s>\n</s>\n')
        for word in vocab[:min(len(vocab),vocab_size)]:
            vocab_file.write(word+'\n')

    print("Vocabulary "+extension+" saved to disk.")

def strip_string(a_string):
    a_string = a_string.replace("\n"," ").replace("\r"," ").replace("\t"," ")

    init_length = len(a_string)
    current_length = init_length - 1
    
    while init_length != current_length:
        init_length = len(a_string)
        a_string.replace("  "," ")
        current_length = len(a_string)
    
    return a_string

def create_headline_datasets():
    with open("../datasets/signalmedia-1m.jsonl", "r") as input_file, open(output_path+'train.content', 'w') as train_contents_file, open(output_path+'train.headlines','w') as train_headlines_file, open(output_path+'test.content','w') as test_contents_file, open(output_path+'test.headlines','w') as test_headlines_file:
        i = 0
        for line in input_file:
            # if i == 10:
            #     break
            json_line = json.loads(line)

            content = strip_string(json_line['content'])
            title = strip_string(json_line['title'])

            print("Iteration "+str(i)+" : "+title)

            if i % 5:
                # Add instance as test data
                test_contents_file.write(json_line['content'])
                test_headlines_file.write(json_line['title'])
            else:
                # Add instance as train data
                train_contents_file.write(json_line['content'])
                train_headlines_file.write(json_line['title'])

            i += 1

def prepare_string(line):
    # print(line)
    line = re.sub(r'[a-zA-Z][a-zA-Z0-9@:%\.\-_\+~#=\/]{2,256}\.[a-z]{2,6}\b([a-zA-Z0-9@:%\-_\+\.~#?&\/=]*)', ' <url> ', line)
    line = re.sub(r'[.]', str(' . '), line)
    line = re.sub(r'[,]', str(' , '), line)
    line = re.sub(r'[!]', str(' ! '), line)
    line = re.sub(r'[?]', str(' ? '), line)
    line = re.sub(r'[:]', str(' : '), line)
    line = re.sub(r'[%]', str(' % '), line)
    line = re.sub(r'[|;"@#^&\*()_\-+={}\[\]\/<>\n\r]', str(' '), line)
    line = re.sub(r'[0-9]+', ' <number> ', line)
    # print(line)
    return strip_string(line)

def replace_entities(source, summary):
    people, organizations, locations = scraping_with_stanford.stanford_extract_entities(source+summary)

    for i in range(len(people)):
        source = re.sub(people[i], ' <person_'+str(i)+'> ', source)
        summary = re.sub(people[i], ' <person_'+str(i)+'> ', summary)

    for i in range(len(organizations)):
        source = re.sub(organizations[i], ' <organization_'+str(i)+'> ', source)
        summary = re.sub(organizations[i], ' <organization_'+str(i)+'> ', summary)

    for i in range(len(locations)):
        source = re.sub(locations[i], ' <location_'+str(i)+'> ', source)
        summary = re.sub(locations[i], ' <locations_'+str(i)+'> ', summary)

    return source, summary

def create_news_summary_dataset():
    texts = []
    summaries = []

    print("Reading data from CSV file...")
    with open('../datasets/news_summary/news_summary.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)

        for row in reader:
            texts.append(row[-1])
            summaries.append(row[-2])

    print("Processing data for learning...")
    with open('../datasets/news_summary/train.source', 'w') as train_source_file, open('../datasets/news_summary/test.source', 'w') as test_source_file, open('../datasets/news_summary/train.summary', 'w') as train_summary_file, open('../datasets/news_summary/test.summary', 'w') as test_summary_file:
        for i in tqdm(range(len(texts))):
        # for i in tqdm(range(10)):
            source = prepare_string(texts[i])
            summary = prepare_string(summaries[i])

            # search_entities = True
            # while search_entities:
            try :
                source, summary = replace_entities(source, summary)

                if i % 5 == 0:
                    test_source_file.write(source+"\n")
                    test_summary_file.write(summary+"\n")
                else:
                    train_source_file.write(source+"\n")
                    train_summary_file.write(summary+"\n")

            # search_entities = False
            except KeyboardInterrupt:
                # search_entities = False
                raise
            except:
                print("The following text was too long for NLTK :")
                print("Source :  "+source)
                print("Summary :  "+summary)
                print("\n\n")
                continue


if __name__ == "__main__":
    # create_headline_datasets()
    # create_vocabularies('content')
    # create_vocabularies('headlines')

    create_news_summary_dataset()
    print()
    create_vocabularies('../datasets/news_summary/','source')
    print()
    create_vocabularies('../datasets/news_summary/','summary')