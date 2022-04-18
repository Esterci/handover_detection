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
        html.P("batch size:"),
        dcc.Slider(
            id="slider",
            min=100,
            max=500,
            value=100,
            marks={
                100: "100",
                144: "144",
                188: "188",
                233: "233",
                277: "277",
                322: "322",
                366: "366",
                411: "411",
                455: "455",
                500: "500",
            },
        ),
        html.P("encodim dimension:"),
        dcc.Slider(id="slider2", min=8, max=10, value=8, step=1),
    ]
)


@app.callback(
    Output("graph", "figure"), Input("slider", "value"), Input("slider2", "value"),
)
def update_bar_chart(batch_size, encoding_dim):

    fig = px.line(
        results_dict[str(batch_size)][str(encoding_dim)],
        x="RMSE",
        y="likelyhood",
        color="target",
    )

    fig.update_layout(autotypenumbers="convert types")
    return fig


app.run_server(debug=True)
