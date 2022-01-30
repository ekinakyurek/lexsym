from absl import flags, app
from src import lexutils
import json


FLAGS = flags.FLAGS

flags.DEFINE_string("lexfile", default=None,
                    help='Seed lexicon path')

flags.DEFINE_string("codefile", default=None,
                    help='qa and code files')

def main(_):
    lexicon, inputs = lexutils.load_lexicon(FLAGS.lexfile, FLAGS.codefile)
    
    filtered_lexicon, swapables = lexutils.filter_lexicon_v2(lexicon, inputs)
    
    lex = {"lexicon": filtered_lexicon, "swapables": swapables}
    
    for k in list(lex["swapables"].keys()):
        if len(lex["swapables"][k]) == 0:
            del lex["swapables"][k]
            del lex["lexicon"][k]

    with open(FLAGS.lexfile.replace(".json", "-swaps.json"), "w") as f:
        json.dump(lex, f)


if __name__ == "__main__":
    app.run(main)
