
import tensorflow as tf

flags = tf.app.flags

# ************** training configuration **************
tf.flags.DEFINE_integer("random_seed", 0, "Random seed for tensorflow and numpy.")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("l2_reg_lambda", 0.005, "L2 regularizaion lambda")
tf.flags.DEFINE_integer("max_epoch", 100, "Number of epochs to train.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs.")
tf.flags.DEFINE_integer("patience", 5, "number of eval steps to be patient for early stopping")

# ************** network configuration **************
tf.flags.DEFINE_integer("memory_size", 20, "Maximum memory size.")
tf.flags.DEFINE_integer("sent_size", 25, "Maximum sentence size.")
tf.flags.DEFINE_integer("embed_size", 300, "Embedding size.")
tf.flags.DEFINE_integer("hidden_size", 300, "Hidden size.")
tf.flags.DEFINE_integer("hops", 1, "Number of hops for the word/sentence attention layer.")

# ************** source configuration **************
tf.flags.DEFINE_string("data_path", "./data/", "Data path")
tf.flags.DEFINE_string("w2v_path", "/qydata/zlict/deep-learning/CNN_sentence/GoogleNews-vectors-negative300.bin", "Pre-trained word vectors path")

FLAGS = tf.flags.FLAGS

