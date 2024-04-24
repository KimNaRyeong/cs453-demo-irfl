from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import argparse

def irfl(bug_report, source_dir):
    """
    1.  Collect all documents (i.e., the bug report and all source files)
    2.  Compute tf-idf vectors of each document
    3.  Compute cosine similarity between each vector
    4.  Rank source files using the similarity
    5.  Report the top five files
    """

    # step1. collect file paths into docs
    docs = [bug_report]
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        # print(dirpath, dirnames, filenames)
        for filename in filenames:
            if filename.endswith('java'):
                docs.append(os.path.join(dirpath, filename))

    # step2. vectorize
    vectorizer = TfidfVectorizer(input="filename", decode_error="ignore", stop_words="english")
    tfidfs = vectorizer.fit_transform(docs)

    # step3. compute cosine similarity
    cos_sim = cosine_similarity(tfidfs)

    # step4. rank
    suspiciousness = zip(docs[1:], cos_sim[0][1:])
    suspiciousness = sorted(suspiciousness, key=lambda x: x[1], reverse=True)

    # step5. report
    for (file, score) in suspiciousness[:5]:
        print("{0:s}: {1:0.3f}".format(file, float(score)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IR Based Fault Localisation')
    parser.add_argument("-b", "--bug_report", required=True,
                        help='the text file that contains the bug report')
    parser.add_argument('-d', "--source_directory", required=True,
                        help='the root of the source directory')

    args = parser.parse_args()
    bug_report = args.bug_report
    source_dir = args.source_directory
    
    irfl(bug_report, source_dir)