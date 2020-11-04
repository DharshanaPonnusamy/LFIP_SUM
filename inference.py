import importlib
from IP_summarizer_prune import IP_summarizer
import argparse
import string
import json
from tqdm import tqdm


def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN)')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, help='input data path')
    parser.add_argument('--save_path', type = str, help = 'result save path')
    parser.add_argument('--pretrained_model', type = str, default='Bert_base',
                        help = 'pre-trained sentence embedding model')
    parser.add_argument('--info_ratio', type=int, default = 80,
                        help = 'information preservation ratio, should be the value between 0 and 100')
    parser.add_argument('--max_sent_len', type=int, default = 35,
                        help = 'Pruning criteria')

    return parser


def main(args):
    module_path = 'pre_trained_models.'

    with open(args.input_path, 'r', encoding='utf-8') as files:
        documents = json.load(files)

    embedding_model_funcs = importlib.import_module(module_path + args.pretrained_model + '.inference')

    model = embedding_model_funcs.load_model()
    print('Pre-trained model is loaded')

    result = []
    for sentence in tqdm(documents['sentences']):
        sentences_ = clean_lines(sentence)

        if sentences_[-1] == '':
            sentences_ = sentences_[:-1]

        if len(sentences_) == 1:
            output = sentence

        else:
            embeddings = embedding_model_funcs.generate_vecs(model, sentences_)
            summarizer = IP_summarizer(sentence, embeddings, threshold=args.info_ratio, max_len=args.max_sent_len)
            output = summarizer.Optimization()
        result.append(list(output))

    documents['summary'] = result
    with open(args.save_path, 'w', encoding='utf-8') as files:
        json.dump(documents, files)


if __name__ == '__main__':
    parser = args()
    args = parser.parse_args()

    if args.info_ratio < 0 or args.info_ratio > 100:
        raise(ValueError('Info ratio should be the value between 0 and 100'))

    if args.pretrained_model not in ['Bert_base', 'SIF', 'USE', 'InferSent']:
        raise(ValueError('Model name should be Bert_base, SIF, USE, or InferSent. Please check the pretrained model name'))

    main(args)



