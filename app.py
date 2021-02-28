from typing import List
import streamlit as st
import altair as alt

from intuitionbuilder import configs
from intuitionbuilder.helpers import Candidate
from intuitionbuilder.utils import make_candidate
from intuitionbuilder.pairs import get_pair


DEFAULT_PAIR_ID = 9
NUM_PAIRS = 9


@st.cache
def load_candidates(pair_id: int,
                    ) -> List[Candidate]:
    """
    load a pair of matrices, and return a Candidate object for each,
    which also contains results of analyses
    """
    candidates = []

    for m in get_pair(pair_id):

        candidate = make_candidate(m)
        candidates.append(candidate)

        assert candidate.matrix.max() < configs.Heatmap.max

        # the two matrices must have same shape, sum
    assert len(set([c.matrix.sum() for c in candidates])) == 1
    assert len(set([c.matrix.shape for c in candidates])) == 1



    return candidates


# sidebar
st.sidebar.title('Hone your intuition')
st.sidebar.write('Improve your intuitive understanding of information-theory by playing a guessing game. '
                 'First, select a pair of matrices, then guess which has the higher mutual information.')

pair_ids = list(range(1, NUM_PAIRS + 1))
scenario_id = st.sidebar.selectbox('Select a pair.',
                                   pair_ids, index=DEFAULT_PAIR_ID - 1)

st.sidebar.write("""
         This visualization is part of a research effort into the distributional structure of nouns in child-directed speech. 
         More info can be found at http://languagelearninglab.org/
     """)

# load data
c1, c2 = load_candidates(scenario_id)
c1: Candidate
c2: Candidate

# color scale
scale = alt.Scale(
    domain=[0, configs.Heatmap.max],
)


# prepare matrix 1 chart
heat_chart1 = alt.Chart(c1.df).mark_rect().encode(
    alt.X('x:O', axis=None),
    alt.Y('y:O', axis=None),
    color=alt.Color('c:Q', scale=scale),
).properties(
    width=configs.Heatmap.width,
    height=configs.Heatmap.width,
)

# prepare matrix 2 chart
heat_chart2 = alt.Chart(c2.df).mark_rect().encode(
    alt.X('x:O', axis=None),
    alt.Y('y:O', axis=None),
    color=alt.Color('c:Q', scale=scale),
).properties(
    width=configs.Heatmap.width,
    height=configs.Heatmap.height,
)


st.header('Which X-Y pairing has higher mutual information?')
col1, col2 = st.beta_columns(2)

is_answered = False
is_correct = False
is_same = False

# candidate 1
with col1:
    st.header('Candidate 1')
    st.altair_chart(heat_chart1)

    if st.button('This one', 1):
        is_answered = True
        if c1.ami > c2.ami:
            is_correct = True
        elif c1.ami == c2.ami:
            is_same = True

# candidate 2
with col2:
    st.header('Candidate 2')
    st.altair_chart(heat_chart2)

    if st.button('This one', 2):
        is_answered = True
        if c1.ami < c2.ami:
            is_correct = True
        elif c1.ami == c2.ami:
            is_same = True


if is_answered:
    col1.write(f'Adjusted I(X;Y) = {c1.ami}')
    col2.write(f'Adjusted I(X;Y) = {c2.ami}')

    if is_correct:
        st.write('Correct!')
    elif is_same:
        st.write('They are the same')
    else:
        st.write('Nope!')

    is_answered = False
    is_correct = False
    is_same = False
