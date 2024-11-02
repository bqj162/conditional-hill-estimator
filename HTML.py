import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import webbrowser
import os

class HTML:
    def __init__(self, plot_list):
        self.plot_list = plot_list

    def generate_HTML(self, filename = "Output.html"):
        html = open(filename, 'w')
        html.write("<html><head></head><body>" + "\n")
        add_js = True
        for plot in self.plot_list:
            inner_html = pyo.plot(
                plot, include_plotlyjs=add_js, output_type='div'
            )
            html.write(inner_html)
            add_js = False

        html.write("</body></html>" + "\n")
        html.close()
        webbrowser.open('file://' + os.path.realpath(filename))

