# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial

# %% [markdown]
# # Load and clean data

# %%
df = pd.read_csv('unicorn-data.csv')


# %%
df.head()


# %%
df = df[(df['G'] >= 41) & (df['MP'] >= 10)] # restrict data to players who played at least half the season and 10 MPG


# %%
df.columns.values


# %%
features = df.columns.values[6:] # extract all features to be used later


# %%
# function to quickly get the part of a dataframe that matches a certain position

def pos_subset(df, pos, colname):
    return df[df[colname] == pos]


# %%
guards = pos_subset(df, 'guard', 'Pos').loc[:, features].values
wings = pos_subset(df, 'wing', 'Pos').loc[:, features].values
bigs = pos_subset(df, 'big', 'Pos').loc[:, features].values

# standardize each subset for PCA

guards = StandardScaler().fit_transform(guards)
wings = StandardScaler().fit_transform(wings)
bigs = StandardScaler().fit_transform(bigs)


# %%
subsets = [pd.DataFrame(guards), pd.DataFrame(wings), pd.DataFrame(bigs)]

result = pd.concat(subsets)

# %% [markdown]
# # Perform PCA

# %%
# function to run PCA on each subset and find the n_components for which the explained variance ratio is above 0.9

def pca_subset(max_components, pos_subset, pos_name):
    
    pca_var_list = []
    
    for n_components in range(2, max_components):
        pca = PCA(n_components = n_components)
        components = pca.fit_transform(pos_subset)
        pca_variance = sum(pca.explained_variance_ratio_)
        pca_var_list.append(pca_variance)
        print("For n_components = {} for {}, explained variance ratio is {}".format(n_components, pos_name, pca_variance))
        
    return pca_var_list, [n for n, i in enumerate(pca_var_list) if i > 0.9][0] + 2


# %%
pca_var_g, var_index_g = pca_subset(21, subsets[0], 'guards')


# %%
pca_var_w, var_index_w = pca_subset(21, subsets[1], 'wings')


# %%
pca_var_b, var_index_b = pca_subset(21, subsets[2], 'bigs')


# %%
print(var_index_g, var_index_w, var_index_b)


# %%
# plot how explained variance ratio changes with n_components among each positional subset

plt.style.use('fivethirtyeight')

pca_fig, ax = plt.subplots()

ax.plot(range(2, 21), pca_var_g, label = 'guards')
ax.plot(range(2, 21), pca_var_w, label = 'wings')
ax.plot(range(2, 21), pca_var_b, label = 'bigs')

ax.set_xlabel('n_components')
ax.set_ylabel('Explained variance ratio')

ax.set_xticks(np.arange(2, 21, 2.0))
ax.legend(loc = 'best')

pca_fig.suptitle("n_components among positional PCA", weight = 'bold', size = 18)

pca_fig.text(x = -0.05, y = -0.08,
    s = '______________________________________________________________',
    fontsize = 14, color = 'grey', horizontalalignment='left', alpha = .3)



pca_fig.savefig('pca-variance.png', dpi = 400, bbox_inches = 'tight')


# %%
# function to run PCA with the n_components found earlier for each positional subset, then return the PCA data for each subset
# function also returns the 5 most important factors for each component for the sake of basic factor loading analysis

def create_pca_pos(pos_df, n_components, main_df, pos_name):
    
    pca = PCA(n_components = n_components)
    
    pos_df = pd.DataFrame(pos_df)
    comp_pos = pca.fit_transform(pos_df)
    pca_df_pos = pd.DataFrame(data = comp_pos, columns = ['pc_%s' %i for i in range(1, pca.n_components_ + 1)])
    
    pca_df_pos['Player'] = pos_subset(main_df, pos_name, 'Pos')['Player'].values
    pca_df_pos['Pos'] = pos_name
    
    most_important = [np.abs(pca.components_[i]).argpartition(-5)[-5:] for i in range(n_components)]
    most_important_names = [features[most_important[i]] for i in range(n_components)]
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_components)}
    df = pd.DataFrame(dic.items())
    
    return pca_df_pos, df


# %%
pca_df_g, factor_loading_g = create_pca_pos(guards, var_index_g, df, 'guard')
pca_df_w, factor_loading_w = create_pca_pos(wings, var_index_w, df, 'wing')
pca_df_b, factor_loading_b = create_pca_pos(bigs, var_index_b, df, 'big')


# %%
# function to separate values in list in the factors column

def create_factor_df(factor_df):
    new_df = pd.DataFrame(factor_df.iloc[:, 1].values.tolist())
    return new_df


# %%
# save factor dfs to be used with explanation in the post

create_factor_df(factor_loading_g).to_csv('guards_factors.csv')
create_factor_df(factor_loading_w).to_csv('wings_factors.csv')
create_factor_df(factor_loading_b).to_csv('bigs_factors.csv')


# %%
# function to get the names and position of each subset

def get_names_pos(df, pos, colname):
    names = pos_subset(df, pos, colname)['Player'].values
    pos = [pos] * len(names)
    
    return names, pos


# %%
guard_names, guard_pos = get_names_pos(df, 'guard', 'Pos')
wing_names, wing_pos = get_names_pos(df, 'wing', 'Pos')
big_names, big_pos = get_names_pos(df, 'big', 'Pos')

names = np.concatenate((guard_names, wing_names, big_names))
pos = np.concatenate((guard_pos, wing_pos, big_pos))

# %% [markdown]
# # Calculate unicorn index

# %%
# get average of the principal components for each position

avg_g = pca_df_g.iloc[:, :-2].mean().values
avg_w = pca_df_w.iloc[:, :-2].mean().values
avg_b = pca_df_b.iloc[:, :-2].mean().values


# %%
# get distance between each player and average values for each position, then return the values

def get_dist_pos(pos_df, avg_pos, pos_names):
    
    euclid = []
    manhat = []
    cheby = []
    
    for index, row in pos_df.iterrows():
        euclid.append(spatial.distance.euclidean(list(row[:-2].values), avg_pos))
        manhat.append(spatial.distance.cityblock(list(row[:-2].values), avg_pos))
        cheby.append(spatial.distance.chebyshev(list(row[:-2].values), avg_pos))
        
    dist_df = pd.DataFrame(list(zip(euclid, manhat, cheby)), columns = ['euclidean', 'manhattan', 'chebyshev'])
    dist_df['player'] = pos_names
    dist_df = dist_df.sort_values(by = ['euclidean'], ascending = False)
    dist_df = dist_df.reset_index(drop = True)
    
    return dist_df


# %%
guards_dist = get_dist_pos(pca_df_g, avg_g, guard_names)
wings_dist = get_dist_pos(pca_df_w, avg_w, wing_names)
bigs_dist = get_dist_pos(pca_df_b, avg_b, big_names)


# %%
# function to plot players by highest distance in each metric for each position

def plot_unique(df, metric, posname, label_height, fname):
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    
    df = df.sort_values(by = metric, ascending = False)
    
    y = df[metric][0:10].values
    x = np.arange(len(y))
    
    ax.bar(x, y, color = 'skyblue', edgecolor = 'white', linewidth = 4.5)
    
    ax.xaxis.set_visible(False)
    
    labels = df['player'].values

    rects = ax.patches
    for rect, label in zip(rects, labels):
        ax.text(rect.get_x() + rect.get_width() / 1.75, label_height, label,
        ha='center', va='bottom', rotation = 'vertical', color = 'black')
        
    fig.suptitle("%s distance for %s" % (metric.capitalize(), posname), size = 18, weight = 'bold', y = 1.005)
    ax.set_title('Highest distance from positional mean', size = 14, fontname = 'Rockwell')
    
    fig.text(x = 0, y = 0.02,
        s = '___________________________________________________________',
        fontsize = 14, color = 'grey', horizontalalignment='left', alpha = .3)

    
    fig.savefig('%s.png' % fname, dpi = 400, bbox_inches = 'tight')


# %%
plot_unique(guards_dist, 'euclidean', 'guards', .7, 'guards_dist_euclid')


# %%
plot_unique(guards_dist, 'manhattan', 'guards', 1.5, 'guards_dist_manhat')


# %%
plot_unique(guards_dist, 'chebyshev', 'guards', .5, 'guards_dist_cheby')


# %%
plot_unique(wings_dist, 'euclidean', 'wings', .7, 'wings_dist_euclid')


# %%
plot_unique(wings_dist, 'manhattan', 'wings', 1.5, 'wings_dist_manhat')


# %%
plot_unique(wings_dist, 'chebyshev', 'wings', .5, 'wings_dist_cheby')


# %%
plot_unique(bigs_dist, 'euclidean', 'bigs', .7, 'bigs_dist_euclid')


# %%
plot_unique(bigs_dist, 'manhattan', 'bigs', 1.5, 'bigs_dist_manhat')


# %%
plot_unique(bigs_dist, 'chebyshev', 'bigs', .5, 'bigs_dist_cheby')


# %%
# function to normalize distances then take the average in order to create the unicorn index

def get_avg_dist(df):
    
    norm = MinMaxScaler().fit_transform(df[['euclidean', 'manhattan', 'chebyshev']])
    df_norm = pd.DataFrame(norm, columns = ['euclidean', 'manhattan', 'chebyshev'])
    df_norm['unicorn index'] = (df_norm['euclidean'] + df_norm['manhattan'] + df_norm['chebyshev']) / 3
    df_norm['player'] = df['player']
    
    df_norm = df_norm.sort_values(by = 'unicorn index', ascending = False)
    df_norm = df_norm.reset_index(drop = True)
    
    return df_norm


# %%
avg_guards = get_avg_dist(guards_dist)
avg_wings = get_avg_dist(wings_dist)
avg_bigs = get_avg_dist(bigs_dist)


# %%
# function to plot unicorn index

def index_plot(df, metric, posname, label_height, fname):
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    
    df = df.sort_values(by = metric, ascending = False)
    
    y = df[metric][0:10].values
    x = np.arange(len(y))
    
    ax.bar(x, y, color = 'skyblue', edgecolor = 'white', linewidth = 4.5)
    
    ax.set_yticks(np.arange(0, 1.2, .2))
    
    ax.xaxis.set_visible(False)
    
    labels = df['player'].values

    rects = ax.patches
    for rect, label in zip(rects, labels):
        ax.text(rect.get_x() + rect.get_width() / 1.75, label_height, label,
        ha='center', va='bottom', rotation = 'vertical', color = 'black')
        
    fig.suptitle("%s for %s" % (metric.capitalize(), posname), size = 18, weight = 'bold', y = 1.005)
    ax.set_title('1 = most unique possible', size = 14, fontname = 'Rockwell')
    
    fig.text(x = 0, y = 0.02,
        s = '___________________________________________________________',
        fontsize = 14, color = 'grey', horizontalalignment='left', alpha = .3)

    
    fig.savefig('%s.png' % fname, dpi = 400, bbox_inches = 'tight')


# %%
index_plot(avg_guards, 'unicorn index', 'guards', .03, 'guards_unicorn_index')


# %%
index_plot(avg_wings, 'unicorn index', 'wings', .03, 'wings_unicorn_index')


# %%
index_plot(avg_bigs, 'unicorn index', 'bigs', .03, 'bigs_unicorn_index')


# %%
subsets = [avg_guards, avg_wings, avg_bigs]

unicorn_df = pd.concat(subsets)
unicorn_df = unicorn_df.reset_index()
unicorn_df['index'] += 1
unicorn_df.rename(columns = {'index': 'positional rank'}, inplace = True)

unicorn_df = unicorn_df.sort_values(by = 'unicorn index', ascending = False)

unicorn_df.to_csv('unicorn_index.csv', index = False)


