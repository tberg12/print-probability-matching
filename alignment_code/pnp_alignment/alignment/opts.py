"""
Created on Jun 26, 2018

@author: kartikgo
"""
import argparse


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    group = parser.add_argument_group("Model")

    group.add_argument("-convk", type=int, default=3, help="Filter size")
    group.add_argument(
        "-mdn_nmix", type=int, default=5, help="Number of MDN mixture components"
    )
    group.add_argument(
        "-zsize", type=int, default=128, help="Latent variable size for VAE"
    )
    group.add_argument("-yemb", type=int, default=-1, help="Embedding for y")
    group.add_argument("-nc", type=int, default=1, help="Number of classes")
    group.add_argument(
        "-bernoulli",
        action="store_true",
        help="use samples for the data instead of the normalized pixel intensities",
    )
    group.add_argument("-mnist", action="store_true", help="build emnist data")
    # group.add_argument('-y_pix', type=int, default=64,
    #                   help='Max y pixel resolution')


def train_opts(parser):
    # Model loading/saving options

    group = parser.add_argument_group("General")
    group.add_argument(
        "-data", default="translation/", help="""Path prefix to the data"""
    )
    group.add_argument(
        "-bincsv",
        default="binarized.csv",
        help="""csv file containing thresholds for binarization""",
    )
    group.add_argument(
        "-no_binarize",
        action="store_true",
        help="""optionally, do not binarize if the images are already binarized""",
    )
    group.add_argument(
        "-aligncsv",
        default="align.csv",
        help="""csv file containing file lists for original and aligned output""",
    )
    group.add_argument("-init_dir", default="", help="""template init directory""")
    group.add_argument("-ink_var", type=float, default=0.02, help="inking param")
    group.add_argument(
        "-interpolate",
        default="area",
        choices=["area", "bicubic", "bilinear", "nearest"],
        help="""Interpolation method.""",
    )
    group.add_argument(
        "-output", default="templates", help="""suffix for output directory"""
    )
    group.add_argument(
        "-rec", default="reconstruction", help="""suffix for reconstruct directory"""
    )
    group.add_argument("-visualize", action="store_true", help="Do visualization")
    group.add_argument(
        "-init_rand",
        action="store_true",
        help="Initialize the templates from random input samples",
    )
    group.add_argument(
        "-init_avg", action="store_true", help="Initialize the templates with avg"
    )
    group.add_argument("-nc", type=int, default=1, help="Number of classes")
    group.add_argument("-y_pix", type=int, default=64, help="Max y pixel resolution")
    group.add_argument(
        "-data_size", type=int, default=1000000, help="No. of images per class"
    )
    group.add_argument(
        "-no_rotn", action="store_true", help="dont do rotation adjustment"
    )
    group.add_argument(
        "-no_shear", action="store_true", help="dont do shear adjustment"
    )
    group.add_argument(
        "-adj_first",
        action="store_true",
        help="apply interpretable latent variables before the latent",
    )
    group.add_argument("-x_pix", type=int, default=42, help="Max y pixel resolution")
    group.add_argument("-max_off", type=int, default=6, help="Max offset")
    group.add_argument("-save_dir", default="saved_models", help="""model dirname""")
    group.add_argument("-nosave", action="store_true", help="do not save the model")
    group.add_argument(
        "-decode_method",
        default="comb",
        choices=["comb", "hmap", "template_emb", "convolve", "both"],
        help="""Decoding method.""",
    )
    group.add_argument(
        "-encode_method",
        default="simple",
        choices=["simple", "template", "shift_template", "shift_img"],
        help="""Decoding method.""",
    )
    group.add_argument(
        "-diff_last",
        action="store_true",
        help="for encoding, take a diff bw conv rep of image and template ",
    )
    group.add_argument(
        "-cluster", action="store_true", help="are meaningful cluster emasures expected"
    )
    # GPU
    group.add_argument(
        "-gpu", default=0, type=int, help="Use CUDA on the listed devices."
    )

    group.add_argument(
        "-seed",
        type=int,
        default=-1,
        help="""Random seed used for the experiments
                       reproducibility.""",
    )

    # Init options
    group = parser.add_argument_group("Initialization")
    group.add_argument(
        "-start_epoch", type=int, default=1, help="The epoch from which to start"
    )
    group.add_argument(
        "-param_init",
        type=float,
        default=0.1,
        help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""",
    )
    group.add_argument(
        "-train_from",
        default="",
        type=str,
        help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""",
    )
    group.add_argument(
        "-init_from",
        default="",
        type=str,
        help="""If initializing from a checkpoint of a smaller model then this is the
                       path to the pretrained model's state_dict.""",
    )

    # Optimization options
    group = parser.add_argument_group("Optimization- Type")
    group.add_argument("-batch_size", type=int, default=64, help="Maximum batch size")
    group.add_argument(
        "-valid_every", type=int, default=1, help="Get valid stats every n epochs"
    )
    group.add_argument("-train_inst", type=int, default=-1, help="Training instances")
    group.add_argument(
        "-epochs", type=int, default=50, help="Number of training epochs"
    )
    group.add_argument(
        "-optim",
        default="sgd",
        choices=["sgd", "adagrad", "adadelta", "adam", "rmsprop", "LBFGS"],
        help="""Optimization method.""",
    )
    group.add_argument(
        "-adj_optim",
        default="sgd",
        choices=["sgd", "adagrad", "adadelta", "adam", "rmsprop", "LBFGS"],
        help="""Optimization method.""",
    )
    group.add_argument(
        "-adagrad_accumulator_init",
        type=float,
        default=0,
        help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """,
    )
    group.add_argument(
        "-max_grad_norm",
        type=float,
        default=5,
        help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""",
    )
    group.add_argument(
        "-dropout",
        type=float,
        default=0.3,
        help="Dropout probability; applied in LSTM stacks.",
    )
    group.add_argument("-lbd", type=float, default=0.0, help="L2 reg on templates")
    group.add_argument("-l2", type=float, default=0.0, help="L2 regularization")
    group.add_argument(
        "-momentum", type=float, default=0.1, help="momentum for optimization"
    )
    group.add_argument(
        "-adam_beta1",
        type=float,
        default=0.9,
        help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""",
    )
    group.add_argument(
        "-adam_beta2",
        type=float,
        default=0.999,
        help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""",
    )
    group.add_argument(
        "-label_smoothing",
        type=float,
        default=0.0,
        help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""",
    )
    group.add_argument(
        "-lmexp",
        type=float,
        default=1.0,
        help="""Exponentiation component for the LM""",
    )
    group.add_argument(
        "-freeze_template_params",
        action="store_true",
        help="Freeze template parameters when using -train_from and evaluating on new book",
    )

    # learning rate
    group = parser.add_argument_group("Optimization- Rate")
    group.add_argument(
        "-learning_rate",
        type=float,
        default=0.001,
        help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""",
    )
    group.add_argument(
        "-adj_learning_rate",
        type=float,
        default=0.1,
        help="""Starting learning rate for lbd's sgd. """,
    )
    group.add_argument(
        "-learning_rate_decay",
        type=float,
        default=0.99,
        help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""",
    )
    group.add_argument(
        "-start_decay_at",
        type=int,
        default=10,
        help="""Start decaying every epoch after and including this
                       epoch""",
    )
    group.add_argument(
        "-tbeam_size", type=int, default=1, help="""Beam size for beam aware training"""
    )
    group.add_argument(
        "-start_checkpoint_at",
        type=int,
        default=0,
        help="""Start checkpointing every epoch after and including
                       this epoch""",
    )
    group.add_argument(
        "-decay_method",
        type=str,
        default="",
        choices=["noam"],
        help="Use a custom decay rate.",
    )
    group.add_argument(
        "-warmup_steps",
        type=int,
        default=4000,
        help="""Number of warmup steps for custom decay.""",
    )

    group = parser.add_argument_group("Logging")
    group.add_argument(
        "-report_every", type=int, default=50, help="Print stats at this interval."
    )


def generate_opts(parser):
    group = parser.add_argument_group("Model")
    group.add_argument("-model", required=True, help="Path to model .pt file")

    group = parser.add_argument_group("Data")

    # group.add_argument('-output', default='pred.txt',
    #                   help="""Path to output the predictions (each line will
    #                   be the decoded sequence""")

    group = parser.add_argument_group("Beam")
    group.add_argument("-beam_size", type=int, default=5, help="Beam size")

    group = parser.add_argument_group("Logging")
    group.add_argument(
        "-verbose",
        action="store_true",
        help="Print scores and predictions for each sentence",
    )
    group.add_argument(
        "-n_best",
        type=int,
        default=1,
        help="""If verbose is set, will output the n_best
                       decoded sentences""",
    )

    group = parser.add_argument_group("Efficiency")
    group.add_argument("-batch_size", type=int, default=30, help="Batch size")
    group.add_argument("-gpu", type=int, default=-1, help="Device to run on")


def add_md_help_argument(parser):
    parser.add_argument(
        "-md",
        action=MarkdownHelpAction,
        help="print Markdown-formatted help text and exit.",
    )


# MARKDOWN boilerplate


# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = "# Options: %s" % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self).start_section("### **%s**" % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append(
            "* **-%s %s** "
            % (action.dest, "[%s]" % action.default if action.default else "[]")
        )
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(["", ""])
        return "\n".join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        **kwargs
    ):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(
            option_strings, dest, nargs=0, help=help, **kwargs
        )

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
