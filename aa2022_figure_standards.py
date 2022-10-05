"""
Standards for figures and plots.

Use the set_rcparams_dynamo routine to get a sane default configuration
for production quality plots.

The defaults should be used for most purposes.


Some guidelines when creating plots:

The default color palette of matplotlib is good for red-green colorblindness,
but may be bad for other types.

For scans where there the variable can be sorted by size,
use the following plot styles:

Variable    Symbol    Color
Large        ^         C0 #1f77b4
 ^           s         C1 #ff7f0e
 |           o         C2 #2ca02c
 v           d         C3 #d62728
Small        v         C4 #d9467bd

Color are robust on both, screen, print, projector.

# TODO:
 - define color list for colorblindness
 - make line style list
 - make plotstyle for including color bars.
    This should ONLY take up real estate on the right hand side of the figure
 - make template for different figures stacked on top of each other
 - legend.handlelength: check more line styles.
"""

# For 5 symbols in a plot
symbol_list_5 = ['^', 's', 'o', 'd', 'v']

# For 4 symbols in a plot
symbol_list_4 = ['^', 's', 'd', 'v']

# 3 symbols in a plot
symbol_list_3 = ['^', 'o', 'v']

# 2 symbols in a plot
symbol_list_2 = ['^', 'v']

# 1 symbol
symbol_list_1 = ['o']


golden_ratio = 0.5 * (1. + 5**0.5)

def set_rcparams_dynamo(myParams, num_cols=1, ls="thin"):
    """
    Half column width figures for revtex, see
    http://publishing.aip.org/authors/preparing-graphics

    Input:
    =======

    myParams...: matplotlib.rcParams, the parameters from matplotlib
    num_cols...: int, number of columnes the figure spans.
                      1 column  = 3.37" by 2.08"
                      2 columns = 6.69" by 4.13"
    ls.........: string, either "thick" or "thin". Defaults to "thin"
    fonts .....: string, either 'small' or 'large'.

    Output:
    =======
    None
    """

    fig_dpi = 300.
    fontsize = 8

    assert(ls in ["thick", "thin"])
    linewidth = 0.75
    if ls == 'thick':
        linewidth *= 2

    assert(num_cols in [1, 2])

    # Define axis size to be used
    if num_cols == 1:
        ax_x0, ax_y0 = 0.2, 0.2
        axes_size = [ax_x0, ax_y0, 0.95 - ax_x0, 0.95 - ax_y0]
    elif num_cols == 2:
        ax_x0, ax_y0 = 0.1, 0.2
        axes_size = [ax_x0, ax_y0, 0.975 - ax_x0, 0.95 - ax_y0]

    # Figure size in inch
    fig_width_in = num_cols * 3.37
    fig_height_in = 3.37 / golden_ratio

    # Figure size and dpi
    myParams['figure.dpi'] = fig_dpi
    myParams['figure.figsize'] = [fig_width_in, fig_height_in]
    myParams['savefig.dpi'] = fig_dpi

    # Figure legend
    myParams['legend.framealpha'] = 1.
    myParams['legend.fancybox'] = False
    myParams['legend.edgecolor'] = 'k'
    myParams['patch.linewidth'] =  0.5 #For legend box borders
    myParams['legend.handlelength'] = 1.45 # Show nice, even
                                           # numbers for different line styles

    # Font and text
    myParams['text.usetex'] = True
    myParams['pdf.fonttype'] = 42
    myParams['font.family'] = 'Times'
    myParams['font.size'] = fontsize
    myParams['axes.labelsize'] = fontsize
    myParams['legend.fontsize'] = fontsize

    # Line size and marker size
    myParams['lines.markersize'] = 3. * linewidth
    myParams['lines.linewidth'] = linewidth

    # Axes thickness
    myParams['axes.linewidth'] = 0.5
    # Enable minor ticks
    myParams['ytick.minor.visible'] = True
    myParams['xtick.minor.visible'] = True
    # Default ticks on both sides of the axes
    myParams["xtick.top"] = True
    myParams["xtick.bottom"] = True
    myParams["ytick.left"] = True
    myParams["ytick.right"] = True
    # All ticks point inward
    myParams["xtick.direction"] = "in"
    myParams["ytick.direction"] = "in"

    return axes_size
