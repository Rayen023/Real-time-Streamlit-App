import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit


# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(
    page_title="Human Activity Classification", 
    page_icon=":running:", 
    layout="wide"
    )


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# ---- READ Dataframe ---- # switch to from Mongo database with secrets file

# Cache the dataframe so it's only loaded once
@st.cache
def get_data():
    df = pd.read_csv('train.csv' , index_col= 0)
    # Add Preprocessing function after importing from the notebook
    return df

df = get_data()
st.session_state["df"] = df

# ---- MAINPAGE ----
st.title(":running: Human Activity Classification")
st.markdown("##")

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")

user = st.sidebar.select_slider(
    "Select the User:",
    options=sorted(df["user"].unique()),
    value = 10
)
# Add a section of day column and filter

df_user_selection = df.query(
    "user == @user"
)

st.dataframe(df , use_container_width= True)

# TOP KPI's
User_activities = df_user_selection["activity"].unique()


activities_per_user = (
    df_user_selection.activity.value_counts()
)
fig_user_activities = px.pie(
    activities_per_user,
    values= list(activities_per_user.values),
    names= list(activities_per_user.index),
    title="<b>Activity per User</b>",
    #color_discrete_sequence=["#0083B8"] * len(activities_per_user),
    template="plotly_white",
)
fig_user_activities.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)



axis = st.sidebar.selectbox(
    label = "Select the Axis:",
    options=['x-axis','y-axis','z-axis'],
)
fig = px.line(df_user_selection, x="timestamp", y=axis, color='activity')

fig.update_layout(title='',
                  plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=(dict(showgrid=False)))


left_column, right_column = st.columns(2)
with left_column:
    st.subheader(f"User {int(user)} Activities")
    st.info(f"{User_activities}")
    left_column.plotly_chart(fig, use_container_width=True)
with right_column:
    st.subheader("Activity Value Counts:")
    right_column.plotly_chart(fig_user_activities, use_container_width=True)




st.markdown("""---""")

height = st.slider(label=' Select graph height' , min_value= 200, max_value= 1000, value= 600)

st.info('Tap Activity to activate or deactivate ')

stats_full = df.groupby(['activity','user'], 
                        as_index=False)[['x-axis','y-axis','z-axis']].std()

fig = px.scatter_3d(data_frame=stats_full,
                    x='x-axis', y='y-axis', z='z-axis',
                    color='activity',
                    opacity=0.25)

fig.update_layout(title='',
                  plot_bgcolor="rgba(0,0,0,0)",
                  height=height,
                  xaxis=(dict(showgrid=False)))

st.plotly_chart(fig , use_container_width= True)

st.markdown("""---""")

colorscales = px.colors.named_colorscales()
scale = st.selectbox( label= 'Select color scale' , options=colorscales,index=4)

stats_full_mean_std = stats_full.groupby('activity')[['x-axis','y-axis','z-axis']].mean()
fig = px.imshow(stats_full_mean_std, x=stats_full_mean_std.columns, y=stats_full_mean_std.index,
                color_continuous_scale= scale
                )
fig.update_layout(title='',
                  plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=(dict(showgrid=False)))

left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(stats_full_mean_std,use_container_width=True )
with right_column:
    st.plotly_chart(fig , use_container_width= True)


st.markdown("""---""")
