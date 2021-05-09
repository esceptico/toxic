from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

from plotly import graph_objects as go
import streamlit as st

from src.toxic.inference import Toxic


POSITIVE_COLOR = '#19E0FF'
NEGATIVE_COLOR = '#E37805'
DESCRIPTION = (
    '<p style="font-size: 0.85em; line-height: 1.5">'
    'Toxic comment classification'
    '<br><strong>Model</strong>: Wide CNN Encoder -> FFNN'
    '<br><strong>Interpretation module</strong>: Integrated gradients'
    '<br><strong>Link</strong>: '
    '<a href="https://github.com/esceptico/toxic">GitHub</a>'
    '</p>'
)
DEFAULT_TEXT = 'вот урод, гореть ему в аду!'


def highlight(
    text: str,
    spans: List[Tuple[int, int]],
    weights: List[float],
    brightness: float = 1.,
    positive_color: str = POSITIVE_COLOR,
    negative_color: str = NEGATIVE_COLOR
):
    colored_text = text
    shift = 0
    for i, ((start, end), weight) in enumerate(zip(spans, weights)):
        token = text[start:end]
        color = positive_color if weight > 0 else negative_color
        alpha = int(abs(weight) * brightness * 255)
        color = f'{color}{alpha:x}'
        token = f'<span style="background-color:{color};">{token}</span>'
        start, end = start + shift, end + shift
        colored_text = colored_text[:start] + token + colored_text[end:]
        shift += len(token) - end + start
    return colored_text


def settings_panel(layout):
    settings_layout = layout.beta_expander('Settings')
    threshold = settings_layout.slider(
        label='Classification threshold',
        min_value=0., max_value=1.,
        step=0.01, value=0.5
    )
    brightness = settings_layout.slider(
        label='Highlight brightness',
        min_value=0., max_value=1.,
        step=0.01, value=1.
    )
    col1, col2 = settings_layout.beta_columns(2)
    positive_color = col1.color_picker('Positive color', POSITIVE_COLOR)
    negative_color = col2.color_picker('Negative color', NEGATIVE_COLOR)
    settings = {
        'brightness': brightness,
        'positive_color': positive_color,
        'negative_color': negative_color,
        'threshold': threshold
    }
    return settings


def sidebar():
    st.sidebar.title('Toxic')
    st.sidebar.write(DESCRIPTION, unsafe_allow_html=True)
    settings = settings_panel(st.sidebar)
    return settings


def plotly_bar_chart(
    data: dict,
    labels: str,
    values: str,
    threshold: float = 0.5
):
    """Plotly bar chart with confidence on X axis and label on Y axis"""
    labels, values = zip(*((item[labels], item[values]) for item in data))
    colors = [POSITIVE_COLOR if item > threshold else 'Grey' for item in values]
    fig = go.Figure(
        [
            go.Bar(
                x=values, y=labels,
                orientation='h',
                marker_color=colors  # noqa
            )
        ],
        layout_xaxis_range=[0, 1]
    )
    fig.add_vline(x=threshold, line_width=2, line_dash="dot", opacity=0.5)
    fig.update_layout(height=300)
    return fig


def body(predict, settings):
    text = st.text_input('Text input', value=DEFAULT_TEXT)
    if text:
        result = predict(text)
        bar = plotly_bar_chart(
            data=result['predicted'],
            labels='class',
            values='confidence',
            threshold=settings['threshold']
        )
        st.plotly_chart(bar, use_container_width=True)

        interpretation = result['interpretation']
        st.subheader('Interpretation results')
        for item in result['predicted']:
            if item['confidence'] >= settings['threshold']:
                st.markdown(f'#### {item["class"]}')
                spans = interpretation['spans']
                weights = interpretation['weights'][item['class']]
                highlighted = highlight(
                    text=text, spans=spans, weights=weights,
                    brightness=settings['brightness'],
                    positive_color=settings['positive_color'],
                    negative_color=settings['negative_color'],
                )
                st.write(highlighted, unsafe_allow_html=True)
        exp = st.beta_expander("Json")
        exp.json(result)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = Toxic.from_checkpoint(path=args.model)

    st.set_page_config(page_title='toxic', layout='wide', page_icon='🤬')
    settings = sidebar()
    body(model.infer, settings)
