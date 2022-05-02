from re import X
from tkinter import Y
from dash import Dash, dcc, html, Input, Output
from matplotlib.pyplot import text
import plotly.graph_objects as go
import plotly.express as px
import pickle as pk

with open("plotly_seed.pkl", "rb") as f:
    results_dict = pk.load(f)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("Interactive plot with custom data source"),
        dcc.Graph(id="graph"),
        html.P("n1:"),
        dcc.Slider(
            id="slider",
            min=20,
            max=200,
            value=20,
            marks={
                20: "20",
                200: "200",
            },
        ),
        html.P("time step:"),
        dcc.Slider(
            id="slider2",
            min=3,
            max=50,
            value=3,
            marks={
                3: "3",
                6: "6",
                10: "10",
                13: '13',
                17: '17',
                20: '20',
                24: '24',
                28: '28',
                31: '31',
                35: '35',
                39: '39',
                42: '49',
                46: '46',
                50: '50',
            },
        ),
    ]
)


@app.callback(
    Output("graph", "figure"), Input("slider", "value"), Input("slider2", "value"),
)
def update_bar_chart(encoding_dim, time_step):

    fig = px.line(
        results_dict['500'][str(encoding_dim)][str(time_step)],
        x="RMSE",
        y="likelyhood",
        color="target",
    )

    fig.update_layout(autotypenumbers="convert types")
    return fig


app.run_server(debug=True)
