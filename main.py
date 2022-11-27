import utils
import argparse
import bert
import cont_bert
import syntax_bert

def main(args):
    cfg = utils.get_config(args['file'])
    
    if cfg['code'] == 'bert':
        """Call BERT code"""
        bert.train_bert(cfg)
    elif cfg['code'] == 'cont_bert':
        cont_bert.cont_BERT(cfg)
    elif cfg['code'] == 'syntax_bert':
        syntax_bert.syntax_BERT(cfg)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Code_mixed Dradician Language Offensive Detection')
    parser.add_argument("file", type=str,  default="train.json"),
    args = parser.parse_args()
    main(args)