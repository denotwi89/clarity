import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import numpy as np
from PIL import Image
import seaborn as sns
import altair as alt
import plotly.graph_objects as go

path = 'https://raw.githubusercontent.com/denotwi89/clarity/'
esgRisk = path + 'portfolios/portfolios/esgRisk/'


def main():
    def remote_css(url):
        st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

    st.set_page_config(layout="wide")

    # remote_css('https://qontigo.com/wp-includes/css/dist/block-library/style.min.css?ver=5.8')
    # load data and cache it using Streamlit cache

    @st.cache
    def load_pfs():
        data = pd.read_csv(path + 'main/portfolio_names.csv')
        return data

    ports = load_pfs()

    @st.cache
    def load_esgRisk_summary():
        data = pd.read_csv(esgRisk + 'summary.csv')
        return data

    ports_esgRisk_summary = load_esgRisk_summary()

    @st.cache
    def load_esgRisk(portfolio_name: str):
        port_id = ports[ports['name'] == portfolio_name]['id'].item()
        esgRisk_data = pd.read_csv(esgRisk + '%s_esgRisk.csv' % port_id)
        return esgRisk_data

    @st.cache
    def get_gics_esgRisk(port_esgRisk_data: pd.DataFrame, bmk_esgRisk_data: pd.DataFrame):
        both = [port_esgRisk_data, bmk_esgRisk_data]
        all_sectors = []
        for each in both:
            each = each.set_index(['GICS Sector', 'GICS Industry Group', 'GICS Industry']).sort_index()
            sectors = each.index.get_level_values('GICS Sector').unique()
            each['Weight in Sector'] = pd.Series(dtype=float)
            for sector in sectors:
                if sector not in all_sectors:
                    all_sectors.append(sector)
                gics_sector = each.xs(sector, level='GICS Sector')
                gics_weight = gics_sector['weight'].sum()
                for asset in gics_sector.index.get_level_values('isin').unique():
                    asset_weight = each.xs(asset, level='isin')['weight'].item()
                    each.at[
                        (sector, slice(None), slice(None), asset), 'Weight in Sector'] = asset_weight / gics_weight

    port_names = tuple(list(ports['name']))
    st.sidebar.subheader("Analysis Settings")
    portfolio = st.sidebar.selectbox(label='Select portfolio', options=port_names)
    benchmark = st.sidebar.selectbox(label='Select benchmark', options=port_names)

    port_esgRisk = load_esgRisk(portfolio)
    bmk_esgRisk = load_esgRisk(benchmark)

    st.markdown("<h1 style='text-align: center'>Clarity AI + Axioma Analytics</h1>", unsafe_allow_html=True)

    portfolio_id = ports[ports['name'] == portfolio]['id'].item()
    benchmark_id = ports[ports['name'] == benchmark]['id'].item()
    ids = [portfolio_id, benchmark_id]

    title = f'{portfolio} vs {benchmark}'
    st.markdown(f"<h2 style='text-align: center'>{title}</h2>", unsafe_allow_html=True)

    my_page = st.sidebar.radio('Contents',
                               ['ESG Risk Summary', 'GICS Sector Breakdown', 'Country Breakdown', 'Asset Distribution'])

    import matplotlib as mpl

    domain = ['Very Low', 'Low', 'Average', 'High', 'Very High']
    #   range_ = ['#A93226', '#E74C3C', '#F1C40F', '#2ECC71', '#16A085']
    range_ = ['#990033', '#fe4500', '#ffdb4d', '#6c3', '#02ab6c']
    break_points = [17, 33, 66, 83, 100]
    color_map = {}
    for color in range(len(domain)):
        color_map.update({domain[color]: range_[color]})

    if my_page == 'ESG Risk Summary':
        def chart0():
            sns.set()
            df = ports_esgRisk_summary
            port_data = df[df.id.isin(ids)].set_index(['name']).transpose()
            table_figures = ['Market Value', 'Asset Coverage %', 'Asset Count', 'TOTAL ESG', 'ENVIRONMENTAL', 'SOCIAL',
                             'GOVERNANCE']
            data_figures = ['TOTAL ESG', 'ENVIRONMENTAL', 'SOCIAL', 'GOVERNANCE']
            table_data = df[df.id.isin(ids)].set_index(['name']).transpose()
            table_data = table_data[table_data.index.isin(table_figures)]
            port_data = port_data[port_data.index.isin(data_figures)]

            port_data = port_data.astype(float)
            st.markdown("<h3 style='text-align: center'>Summary</h3>", unsafe_allow_html=True)

            formatdict = {}
            table_transpose = table_data.transpose()
            column = [x for x in table_figures if '%' in x]
            for col in column:
                table_transpose[col] = table_transpose[col] / 100
                formatdict[col] = "{:.2%}"

            first_bar = port_data[portfolio]
            first_bar_label = portfolio
            first_bar_color = '#0072ce'
            second_bar = port_data[benchmark]
            second_bar_label = benchmark
            second_bar_color = '#6c3'
            labels = port_data.index
            width = 0.4  # the width of the bars
            plot_title = 'ESG Risk Scores'
            title_size = 14

            # Plot figure
            fig, ax = plt.subplots(figsize=(12.5, 3), facecolor=(.94, .94, .94))
            plt.tight_layout()

            # Plot double bars
            y = np.arange(len(labels))  # the label locations
            ax.barh(y + width / 2, first_bar, width, label=first_bar_label, color=first_bar_color)
            ax.barh(y - width / 2, second_bar, width, label=second_bar_label, color=second_bar_color)

            # Format ticks
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

            # Create labels
            rects = ax.patches
            for rect in rects:
                # Get X and Y placement of label from rect.
                x_value = rect.get_width()
                y_value = rect.get_y() + rect.get_height() / 2
                space = 5
                ha = 'left'
                if x_value < 0:
                    space *= -1
                    ha = 'right'
                label = '{:,.0f}'.format(x_value)
                plt.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(space, 0),
                    textcoords='offset points',
                    va='center',
                    ha=ha)

            # Set y-labels and legend
            ax.set_yticklabels(labels)
            ax.legend()

            # To show each y-label, not just even ones
            plt.yticks(np.arange(min(y), max(y) + 1, 1.0))

            # Adjust subplots
            plt.subplots_adjust(left=0.165, top=.8, right=0.7)

            # Set title
            title = plt.title(plot_title, pad=20, fontsize=title_size)
            title.set_position([.625, 2])
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # Set subtitle
            tform = ax.get_xaxis_transform()

            image_file = path + 'portfolios/graphs/%s_%s.png' % (portfolio_id, benchmark_id)
            plt.savefig(image_file, facecolor=(.94, .94, .94))
            image = Image.open(image_file)
            st.image(image)
            st.dataframe(table_transpose.style.format(formatdict))

        chart0()

    elif my_page == 'Asset Distribution':
        def chart1():

            port_esgRisk = load_esgRisk(portfolio)
            bmk_esgRisk = load_esgRisk(benchmark)

            port_esgRisk = port_esgRisk[(port_esgRisk['ESG'] != 0) & (port_esgRisk['ESG relevance'] >= 25)]
            bmk_esgRisk = bmk_esgRisk[(bmk_esgRisk['ESG'] != 0) & (bmk_esgRisk['ESG relevance'] >= 25)]

            st.markdown("<h3 style='text-align: center'>ESG Risk Scores vs Data Relevance</h3>", unsafe_allow_html=True)

            brush1 = alt.selection(type='interval')
            p = alt.Chart(port_esgRisk.dropna(subset=['ESG'])).mark_circle(size=100, opacity=0.5, stroke='black',
                                                                           strokeWidth=0.2).encode(
                x=alt.X('ESG relevance', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(gridColor='white')),
                y=alt.Y('ESG', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(gridColor='white')),
                tooltip=['company', 'isin', 'ESG', 'ESG relevance'],
                color=alt.Color('ESG Score (Worst to Best)',
                                scale=alt.Scale(domain=domain, range=range_))).properties(
                title=portfolio, width=600, padding={'top': 30, 'bottom': 30, 'left': 30,
                                                     'right': 30}).add_selection(brush1)
            b = alt.Chart(bmk_esgRisk.dropna(subset=['ESG'])).mark_circle(size=100, opacity=0.5, stroke='black',
                                                                          strokeWidth=0.2).encode(
                x=alt.X('ESG relevance', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(gridColor='white')),
                y=alt.Y('ESG', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(gridColor='white')),
                tooltip=['company', 'isin', 'ESG', 'ESG relevance'],
                color=alt.Color('ESG Score (Worst to Best)',
                                scale=alt.Scale(domain=domain, range=range_))).properties(
                title=benchmark, width=600, padding={'top': 30, 'bottom': 30, 'left': 30,
                                                     'right': 30}).add_selection(brush1)
            col1, col2 = st.columns([2, 2])
            with col1:
                st.altair_chart(p.configure(background='#f0f0f0'))
            with col2:
                st.altair_chart(b.configure(background='#f0f0f0'))

        def chart2():
            st.markdown("<h3 style='text-align: center'>Pillar Scores Distributions</h3>", unsafe_allow_html=True)
            port_esgRisk = load_esgRisk(portfolio)
            bmk_esgRisk = load_esgRisk(benchmark)

            port_esgRisk = port_esgRisk[(port_esgRisk['ESG'] != 0) & (port_esgRisk['ESG relevance'] >= 25)]
            bmk_esgRisk = bmk_esgRisk[(bmk_esgRisk['ESG'] != 0) & (bmk_esgRisk['ESG relevance'] >= 25)]
            port = port_esgRisk.rename(columns={'ESG': 'TOTAL'}).dropna(subset=['TOTAL'])
            bmk = bmk_esgRisk.rename(columns={'ESG': 'TOTAL'}).dropna(subset=['TOTAL'])
            # range_ = ['#e60122', '#fe4500', '#ffdb4d', '#6c3', '#02ab6c']
            stops = [  # alt.GradientStop(color='#990033', offset=0),
                # alt.GradientStop(color='#f90', offset=0.17),
                # alt.GradientStop(color='#ffcc00', offset=0.33),
                # alt.GradientStop(color='#ffff00', offset=0.50),
                alt.GradientStop(color='#6c3', offset=0.5),
                # alt.GradientStop(color='#0072ce', offset=0.83),
                alt.GradientStop(color='#0072ce', offset=1)]
            cols = ['ENVIRONMENTAL', 'SOCIAL', 'GOVERNANCE']
            port['env_mean'] = np.mean(port['ENVIRONMENTAL'])
            p = alt.Chart(port, title=portfolio).transform_fold(cols, as_=['Pillar', 'value']).transform_density(
                density='value', groupby=['Pillar'], extent=[0, 100]).mark_area(
                line={'color': '#0072ce', 'size': 1},
                color=alt.Gradient(gradient='linear',
                                   stops=stops, x1=0, x2=1,
                                   y1=0, y2=0)).encode(
                alt.X('value:Q', axis=alt.Axis(gridColor='white')),
                alt.Y('density:Q', axis=alt.Axis(gridColor='white')),
                alt.Row('Pillar:N')).configure(
                background='#f0f0f0').properties(
                height=75, padding=
                {'top': 30, 'bottom': 30,'left': 30, 'right': 30}
            ).configure_title(anchor='middle')
            b = alt.Chart(bmk).transform_fold(cols, as_=['Pillar', 'value']).transform_density(
                density='value', groupby=['Pillar'], extent=[0, 100]).mark_area(
                line={'color': '#0072ce', 'size': 1},
                color=alt.Gradient(gradient='linear',
                                   stops=stops, x1=0, x2=1,
                                   y1=0, y2=0)).encode(
                alt.X('value:Q', axis=alt.Axis(gridColor='white')),
                alt.Y('density:Q', axis=alt.Axis(gridColor='white')),
                alt.Row('Pillar:N')).properties(height=75, title=benchmark, padding={'top': 30, 'bottom': 30,
                                                                                     'left': 30,
                                                                                     'right': 30}).configure(
                background='#f0f0f0')
            col1, col2 = st.columns([2, 2])
            with col1:
                st.altair_chart(p)
            with col2:
                st.altair_chart(b.configure_title(anchor='middle'))

        def chart6():
            port_esgRisk = load_esgRisk(portfolio)
            bmk_esgRisk = load_esgRisk(benchmark)
            st.markdown("<h3 style='text-align: center'>ESG Scores Distribution</h3>", unsafe_allow_html=True)
            tables = {'por': {'df': port_esgRisk,
                              'name': portfolio,
                              'bar_color': '#0072ce',
                              'line_color': '#6c3'},
                      'bmk': {'df': bmk_esgRisk,
                              'name': benchmark,
                              'bar_color': '#f90',
                              'line_color': '#0072ce'}}

            col1, col2 = st.columns([2, 2])
            fig = []

            for key, val in tables.items():
                port = val['df']
                port = port[port.ESG != 0]
                base = alt.Chart(port)
                bar = base.mark_bar().encode(
                    x=alt.X('ESG:Q', axis=alt.Axis(title='Security ESG Scores with Median', gridColor='white'),
                            bin=alt.BinParams(minstep=5, maxbins=20)),
                    y=alt.Y('count(ESG):Q', axis=alt.Axis(title='', gridColor='white')),
                    tooltip=['count(ESG):Q'],
                    size=alt.value(20),
                    color=alt.value(val['bar_color']),
                    # And if it's not true it sets the bar steelblue.)
                ).properties(width=500)

                rule = base.mark_rule(color=val['line_color']).encode(
                    x='median(ESG):Q',
                    size=alt.value(5))
                figure = bar + rule
                fig.append(figure.configure(background='#f0f0f0').properties(
                    title=val['name'], height=300,
                    padding={'top': 30, 'bottom': 30,
                             'left': 30, 'right': 30}))
            port_esgRisk = port_esgRisk.rename(columns={'company': 'Company'}).sort_values(by=['ESG'], ascending=False)
            bmk_esgRisk = bmk_esgRisk.rename(columns={'company': 'Company'}).sort_values(by=['ESG'], ascending=False)
            columns = ['Company', 'weight', 'ESG', 'ENVIRONMENTAL', 'SOCIAL', 'GOVERNANCE']
            with col1:
                st.altair_chart(fig[0])
            with col2:
                st.altair_chart(fig[1])

            st.markdown("<h3 style='text-align: center'>Top & Bottom 10 Assets for ESG Risk</h3>", unsafe_allow_html=True)
            table_port = port_esgRisk[(port_esgRisk['ESG'] != 0) & (port_esgRisk['Company']) & (port_esgRisk['weight'] != 0)][columns]
            #st.dataframe(table_port)
            table_bmk = bmk_esgRisk[(bmk_esgRisk['ESG'] != 0) & (bmk_esgRisk['Company']) & (port_esgRisk['weight'] != 0)][columns]
            formatdict = {}
            for column in port_esgRisk.columns.to_list():
                if column == 'weight':
                    table_port['weight'] = table_port['weight'] / 100
                    table_bmk['weight'] = table_bmk['weight'] / 100
                    formatdict[column] = "{:.2%}"
                elif column != 'Company':
                    formatdict[column] = "{:.0f}"
                else:
                    pass
            col3, col4 = st.columns([2, 2])
            with col3:
                table_port = table_port.sort_values(by='ESG', ascending=False)
                st.dataframe(table_port.head(n=10).style.format(formatdict))
                st.dataframe(table_port.tail(n=10).style.format(formatdict))
            with col4:
                table_bmk = table_bmk.sort_values(by='ESG', ascending=False)
                st.dataframe(table_bmk.head(n=10).style.format(formatdict))
                st.dataframe(table_bmk.tail(n=10).style.format(formatdict))
        chart6()
        chart1()
        chart2()

    #   print(color_map)
    elif my_page == 'Country Breakdown':

        def chart5(name):
            port_id = ports[ports['name'] == name]['id'].item()
            per_country = pd.read_csv(path + 'portfolios/country/%s_country.csv' % port_id)
            per_country = per_country[per_country['Country Weight %'] != 0]
            fig = px.scatter_geo(per_country, locations="Country", color="ESG Rating",
                                 hover_name="Country", size="Country Weight %", color_discrete_map=color_map,
                                 projection="miller", size_max=35, opacity=0.6, title=name)
            fig.update_layout(width=550, margin={"r": 0, "t": 25, "l": 0, "b": 0},
                              title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
            st.plotly_chart(fig)
            per_country.rename(columns={'Country': name}, inplace=True)
            per_country = per_country.set_index(name)
            return per_country

        def chart7():
            st.markdown("<h3 style='text-align: center'>ESG Scores by Country of Headquarters</h3>",
                        unsafe_allow_html=True)

            tables = {'por': {'df': port_esgRisk,
                              'name': portfolio,
                              'bar_color': '#0072ce',
                              'line_color': '#6c3'},
                      'bmk': {'df': bmk_esgRisk,
                              'name': benchmark,
                              'bar_color': '#f90',
                              'line_color': '#0072ce'}}
            layout = go.Layout(
                autosize=False,
                width=1200,
                height=500,
                margin=go.layout.Margin(
                    pad=20
                ))
            fig = go.Figure(layout=layout)

            port_x = port_esgRisk[port_esgRisk['Norm Weight'] != 0].country.unique().tolist()
            port_y = []
            for country in port_x:
                weight = port_esgRisk[port_esgRisk['country'] == country]['Norm Weight'].sum()
                port_y.append(port_esgRisk[port_esgRisk['country'] == country]['ESG Risk Score Contr.'].sum() / weight)

            bmk_x = bmk_esgRisk[bmk_esgRisk['Norm Weight'] != 0].country.unique().tolist()
            bmk_y = []
            for country in bmk_x:
                weight = bmk_esgRisk[bmk_esgRisk['country'] == country]['Norm Weight'].sum()
                bmk_y.append(bmk_esgRisk[bmk_esgRisk['country'] == country]['ESG Risk Score Contr.'].sum() / weight)

            fig.add_trace(go.Bar(x=port_x,
                                 y=port_y,
                                 name=portfolio,
                                 marker_color='#0072ce'
                                 ))
            fig.add_trace(go.Bar(x=bmk_x,
                                 y=bmk_y,
                                 name=benchmark,
                                 marker_color='#6c3'
                                 ))

            fig.update_layout(
                xaxis=dict(title='%s vs %s ESG Scores per Country' % (portfolio, benchmark),
                           categoryorder='total descending'),
                yaxis=dict(
                    title='ESG Score',
                ),
                legend=dict(
                    bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)'
                ),
                barmode='group',
                bargap=0.35,  # gap between bars of adjacent location coordinates.
                bargroupgap=0.05  # gap between bars of the same location coordinate.
            )
            st.plotly_chart(fig)

            st.markdown("<h3 style='text-align: center'>Weights by Country of Headquarters</h3>",
                        unsafe_allow_html=True)

            tables = {'por': {'df': port_esgRisk,
                              'name': portfolio,
                              'bar_color': '#0072ce',
                              'line_color': '#6c3'},
                      'bmk': {'df': bmk_esgRisk,
                              'name': benchmark,
                              'bar_color': '#f90',
                              'line_color': '#0072ce'}}
            col1, col2 = st.columns([2, 2])
            fig = []
            for key, val in tables.items():
                port = val['df']
                port = port[port.ESG != 0]
                print(port)
                base = alt.Chart(port)
                bar = base.mark_bar().encode(
                    x=alt.X('country:N', axis=alt.Axis(title='Country of Headquarters', gridColor='white'), sort='-y'),
                    y=alt.Y('count(country):Q', axis=alt.Axis(title='Asset Count', gridColor='white')),
                    tooltip=['country:N', 'count(country):Q'],
                    size=alt.value(10),
                    color=alt.value(val['bar_color']),
                    # And if it's not true it sets the bar steelblue.)
                ).properties(width=500)
                fig.append(bar.configure(background='#f0f0f0').properties(
                    title=val['name'], height=300,
                    padding={'top': 30, 'bottom': 30,
                             'left': 30, 'right': 30}))
            with col1:
                st.altair_chart(fig[0])
            with col2:
                st.altair_chart(fig[1])

        st.markdown("<h3 style='text-align: center'>Company Headquarter Locations</h3>", unsafe_allow_html=True)
        formatdict = {}

        col1, col2 = st.columns([2, 2])
        with col1:
            port_country = chart5(portfolio)
        with col2:
            bmk_country = chart5(benchmark)

        chart7()

        port_country['Country Weight %'] = port_country['Country Weight %'] / 100
        bmk_country['Country Weight %'] = bmk_country['Country Weight %'] / 100

        for column in port_country.columns.to_list():
            if '%' in column:
                formatdict[column] = "{:.2%}"
            if 'Contr.' in column:
                formatdict[column] = "{:.2f}"
            if column == 'ESG':
                formatdict[column] = "{:.0f}"

        col1, col2 = st.columns([2, 2])
        with col1:
            st.dataframe(port_country.style.format(formatdict))
        with col2:
            st.dataframe(bmk_country.style.format(formatdict))

    elif my_page == 'GICS Sector Breakdown':
        def chart3():
            st.markdown("<h3 style='text-align: center'>ESG Scores by Sector</h3>", unsafe_allow_html=True)
            port = port_esgRisk.set_index(['GICS Sector', 'GICS Industry Group', 'GICS Industry', 'isin']).sort_index()
            bmk = bmk_esgRisk.set_index(['GICS Sector', 'GICS Industry Group', 'GICS Industry', 'isin']).sort_index()
            gics = pd.read_csv(path + 'portfolios/gics.csv')
            gics.fillna('Unclassified', inplace=True)
            gics = gics.set_index(['GICS Sector', 'GICS Industry Group', 'GICS Industry', 'isin']).sort_index()
            sector_names = gics.index.get_level_values('GICS Sector').unique().tolist()
            port_sector_names = port.index.get_level_values('GICS Sector').unique().tolist()
            bmk_sector_names = bmk.index.get_level_values('GICS Sector').unique().tolist()
            df = pd.DataFrame(columns=['GICS Sector', 'Portfolio Weight', 'Portfolio Norm Weight', 'Benchmark Weight',
                                       'Benchmark Norm Weight', 'Portfolio ESG', 'Port ESG Score Contribution',
                                       'Benchmark ESG', 'Bmk ESG Score Contribution'])
            i = 0
            for sector in sector_names:
                df_row = {'GICS Sector': sector}
                if sector in port_sector_names:
                    port_sector = port.xs(sector, level='GICS Sector')
                    df_row.update({'Portfolio Weight': port_sector['weight'].sum(),
                                   'Portfolio ESG': port_sector['ESG Risk Sector Score'].sum(),
                                   'Portfolio Norm Weight': port_sector['Norm Weight'].sum(),
                                   'Port ESG Score Contribution': port_sector['ESG Risk Score Contr.'].sum()})
                else:
                    df_row.update({'Portfolio Weight': float(0),
                                   'Portfolio ESG': float(0),
                                   'Portfolio Norm Weight': float(0),
                                   'Port ESG Score Contribution': float(0)}
                                  )
                if sector in bmk_sector_names:
                    bmk_sector = bmk.xs(sector, level='GICS Sector')
                    df_row.update({'Benchmark Weight': bmk_sector['weight'].sum(),
                                   'Benchmark ESG': bmk_sector['ESG Risk Sector Score'].sum(),
                                   'Benchmark Norm Weight': bmk_sector['Norm Weight'].sum(),
                                   'Bmk ESG Score Contribution': bmk_sector['ESG Risk Score Contr.'].sum()})
                else:
                    df_row.update({'Benchmark Weight': float(0),
                                   'Benchmark ESG': float(0),
                                   'Benchmark Norm Weight': float(0),
                                   'Bmk ESG Score Contribution': float(0)})
                ext_df_row = pd.DataFrame(df_row, index=[i])
                # print(ext_df_row)
                ext_df_row.fillna(0, inplace=True)
                df = df.append(ext_df_row)
                df['Active_Weight'] = df['Portfolio Weight'] - df['Benchmark Weight']
                i += 1

            chart = alt.Chart(df).mark_bar(  # line={'color': '#0072ce', 'size': 2},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#6c3', offset=0),
                           alt.GradientStop(color='#0072ce', offset=0.5)],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0
                )).encode(
                x=alt.X("Active_Weight:Q", axis=alt.Axis(title='Active Weight (%)', gridColor='white')),
                y=alt.Y('GICS Sector:N', stack=None, axis=alt.Axis(title='', gridColor='white'), sort='x'),
                tooltip=['GICS Sector:N', 'Active_Weight:Q'],
                size=alt.value(15),
                color=alt.condition(
                    alt.datum.Active_Weight > 0,
                    alt.value("#0072ce"),  # The positive color
                    alt.value("#f90")  # The negative color
                )).configure(
                background='#f0f0f0').properties(
                title='Active Weight by Sector',
                padding={'top': 30, 'bottom': 30,
                         'left': 200, 'right': 200}, width=1200, height=400)

            port['name'] = portfolio
            bmk['name'] = benchmark
            frames = [port.reset_index(), bmk.reset_index()]
            df2 = pd.concat(frames)
            #print(df2.columns.tolist())
            domain = [portfolio, benchmark]
            range_ = ["#0072ce", '#6c3']
            chart2 = alt.Chart(df2).mark_bar().encode(
                x=alt.X('name:N', axis=None),
                y=alt.Y("sum(ESG Risk Sector Score):Q", stack=None, axis=alt.Axis(title='ESG Score', gridColor='white'), sort='x'),
                tooltip=['GICS Sector:N', 'sum(ESG Risk Sector Score):Q'],
                size=alt.value(15),
                color=alt.Color('name:N', scale=alt.Scale(domain=domain, range=range_)),
                column=alt.Column('GICS Sector:N', header=alt.Header(
                    labelOrient="bottom", labelPadding=-10, title=None, labelAnchor='middle', orient='bottom',
                    titleBaseline='top', titleLineHeight=50
                ))).configure(
                background='#f0f0f0').properties(
                title='Portfolio & Benchmark ESG Scores by Sector',
                padding={'top': 50, 'bottom': 30,
                         'left': 50, 'right': 50}, width=60, height=400).configure_title(align='center', anchor='middle')
            st.altair_chart(chart2)
            st.markdown("<h3 style='text-align: center'>Active Weights</h3>", unsafe_allow_html=True)
            st.altair_chart(chart)
            return df

        chart3_df = chart3()

        def chart4():
            df = ports_esgRisk_summary

            table_figures = ['Market Value', 'Asset Coverage %', 'Asset Count', 'TOTAL ESG', 'ENVIRONMENTAL', 'SOCIAL',
                             'GOVERNANCE']
            table_data = df[df.id.isin(ids)].set_index(['name']).transpose()
            table_data = table_data[table_data.index.isin(table_figures)]
            table_transpose = table_data.transpose()
            import plotly.graph_objects as go
            st.markdown("<h3 style='text-align: center'>Sector Contributions to ESG Score</h3>", unsafe_allow_html=True)
            domain = ['Comm Svcs', 'Consumer Disc.', 'Consumer Staples', 'Energy',
                      'Financials', 'Health Care', 'Industrials', 'IT', 'Materials',
                      'Real Estate', 'Unclassified', 'Utilities']
            range2_ = ['#F4D03F ', '#DC7633', '#F39C12', '#5D6D7E', '#2471A3', '#2ECC71', '#7E5109', '#3498DB',
                       '#E74C3C', '#F1948A', '#717D7E', '#45B39D']

            chart3_df['Portfolio Normalized Weight'] = chart3_df['Portfolio Norm Weight'] * 100
            chart3_df['Benchmark Normalized Weight'] = chart3_df['Benchmark Norm Weight'] * 100
            chart3_df['Port ESG Score Contribution'] = chart3_df['Port ESG Score Contribution'] * 100 / \
                                                       table_transpose.loc[portfolio, 'TOTAL ESG']
            chart3_df['Bmk ESG Score Contribution'] = chart3_df['Bmk ESG Score Contribution'] * 100 / \
                                                       table_transpose.loc[benchmark, 'TOTAL ESG']

            port_fig = px.scatter(chart3_df, y="Port ESG Score Contribution",  x="Portfolio ESG",
                                  size='Portfolio Normalized Weight', color="GICS Sector",
                                  hover_name="GICS Sector", log_x=False, size_max=60,
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  title=portfolio)  # px.colors.qualitative.Pastel

            bmk_fig = px.scatter(chart3_df, x="Benchmark ESG", y="Bmk ESG Score Contribution",
                                 size='Benchmark Normalized Weight', color="GICS Sector",
                                 hover_name="GICS Sector", log_x=False, size_max=60,
                                 color_discrete_sequence=px.colors.qualitative.Pastel, title=benchmark)
            # x_min = min(min(chart3_df["Portfolio ESG"], chart3_df["Benchmark ESG"])
            y_max = max(max(chart3_df["Port ESG Score Contribution"]), max(chart3_df["Bmk ESG Score Contribution"]))
            port_fig.update_xaxes(range=[0, 100])
            port_fig.update_yaxes(range=[0, y_max + 10])
            port_fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                   legend=dict(bgcolor='#f0f0f0', orientation="h",
                                               yanchor="bottom",
                                               y=-.7,
                                               xanchor="right",
                                               x=1),
                                   plot_bgcolor='#f0f0f0', margin_pad=50, height=600, width=600)
            port_fig.update_traces(marker=dict(line=dict(width=1,
                                                         color='grey')),
                                   selector=dict(mode='markers'))
            port_fig.add_vline(x=table_transpose.loc[portfolio, 'TOTAL ESG'], line_color='#0072ce')
            bmk_fig.update_xaxes(range=[0, 100])
            bmk_fig.update_yaxes(range=[0, y_max + 10])
            bmk_fig.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                   legend=dict(bgcolor='#f0f0f0', orientation="h",
                                               yanchor="bottom",
                                               y=-.7,
                                               xanchor="right",
                                               x=1),
                                   plot_bgcolor='#f0f0f0', margin_pad=50, height=600, width=600)
            bmk_fig.update_traces(marker=dict(line=dict(width=1,
                                                        color='grey')),
                                  selector=dict(mode='markers'))
            bmk_fig.add_vline(x=table_transpose.loc[benchmark, 'TOTAL ESG'], line_color='#0072ce')

            col1, col2 = st.columns([2, 2])
            with col1:
                # st.altair_chart(port_bubble)
                st.plotly_chart(port_fig)
            with col2:
                # st.altair_chart(bmk_bubble)
                st.plotly_chart(bmk_fig)
            st.caption("Size of bubbles indicate the weight of the sector")

        chart4()
        chart3_df.set_index('GICS Sector', inplace=True)

        chart3_df.drop(columns=['Portfolio Norm Weight', 'Benchmark Norm Weight'], inplace=True)
        weights = []
        non_weights = []
        cont = []
        chart3_df['Port ESG Score Contribution'] = chart3_df['Port ESG Score Contribution'] / 100
        chart3_df['Bmk ESG Score Contribution'] = chart3_df['Bmk ESG Score Contribution'] / 100
        for column in chart3_df.columns.tolist():
            if 'Weight' in column:
                weights.append(column)
            elif 'Contribution' in column:
                cont.append(column)
            else:
                non_weights.append(column)
        formatdict = {}

        for col in weights:
            chart3_df[col] = chart3_df[col] / 100
            formatdict[col] = "{:.2%}"
        for col in non_weights:
            formatdict[col] = "{:.0f}"
        for col in cont:
            formatdict[col] = "{:.2%}"
        st.dataframe(chart3_df.style.format(formatdict))

    else:
        pass


if __name__ == "__main__":
    main()
