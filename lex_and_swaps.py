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

    with open(FLAGS.lexfile.replace(".json", "-swaps.json"), "w") as f:
        json.dump({"lexicon": filtered_lexicon, "swapables": swapables}, f)


if __name__ == "__main__":
    app.run(main)
