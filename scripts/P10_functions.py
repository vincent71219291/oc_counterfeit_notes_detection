import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    silhouette_score
)


@plt.rc_context({'axes.titleweight': 'bold'})
def check_normality_of_residuals(model, axes=None):
    residuals = model.resid
    
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    sns.histplot(
        residuals, stat='density', kde=True, label='Résidus', ax=axes[0]
    )

    x = np.linspace(residuals.min(), residuals.max(), 50)
    y = stats.norm.pdf(x, residuals.mean(), residuals.std())
    sns.lineplot(x=x, y=y, color='red', label='Normale', ax=axes[0])

    sm.qqplot(residuals, line='s', ax=axes[1])
    
    axes[0].set_title('Distribution des résidus')
    axes[1].set_title('Q-Q plot')


@plt.rc_context({'axes.titleweight': 'bold'})
def plot_residuals_vs_fitted(model, ax=None):
    residuals = model.resid
    fitted = model.fittedvalues
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    sns.scatterplot(x=fitted, y=residuals, ax=ax)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('Residual')
    ax.set_title('Résidus vs. valeurs prédites')


def get_studentized_residuals_wls(wls_model):
    res = sm.OLS(wls_model.model.wendog, wls_model.model.wexog).fit()
    infl = res.get_influence()
    return infl.resid_studentized_internal


@plt.rc_context({'axes.titleweight': 'bold'})
def plot_studentized_resids_vs_fitted_wls(wls_model, ax=None):
    studentized_resids = get_studentized_residuals_wls(wls_model)
    fitted = wls_model.fittedvalues

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    sns.scatterplot(x=fitted, y=studentized_resids, ax=ax)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('Studentized residual')
    ax.set_title('Résidus studentisés vs. valeurs prédites')


def get_unusual_and_influencial_obs_wls(wls_model, data):
    df = data.copy()
    n, p = wls_model.model.exog.shape
    
    res = sm.OLS(wls_model.model.wendog, wls_model.model.wexog).fit()
    infl = res.get_influence()
    
    df['fitted'] = wls_model.fittedvalues
    
    # on identifie les valeurs atypiques de x avec les leviers
    leverages = infl.hat_matrix_diag
    lev_threshold = 2 * p / n
    df['has_unusual_x_values'] = leverages > lev_threshold
    
    # on identifie les valeurs atypiques de y avec les résidus standardisés    
    studentized_resids = infl.resid_studentized_internal
    df['studentized_resids'] = studentized_resids
    df['has_unusual_y_value'] = np.abs(studentized_resids) > 3
    
    # on identifie les valeurs influentes avec la dffits (difference in fits)
    dffits = infl.dffits[0]
    dffits_threshold = 2 * np.sqrt((p + 1) / (n - p - 1))
    df['is_influencial_dffits'] = np.abs(dffits) > dffits_threshold
    
    # on identifie les valeurs influentes avec la distance de Cook
    cooks_d = infl.cooks_distance[0]
    cooks_d_threshold = 4 / (n - p)
    df['is_influencial_cooks_d'] = cooks_d > cooks_d_threshold
    
    df['is_unusual_and_influencial'] = (
        (df['has_unusual_x_values'] | df['has_unusual_y_value'])
        & (df['is_influencial_dffits'] | df['is_influencial_cooks_d'])
    )
    
    return df


def wls_formula_backward_selection(data, endog_name, weights, alpha=0.05): 
    exog_names = list(data.columns)
    exog_names.remove(endog_name)
    i = 0
    
    while True:
        formula = endog_name + ' ~ ' + ' + '.join(exog_names)
        res_wls = smf.wls(formula, data=data, weights=weights).fit()
        pvalues = res_wls.pvalues.drop('Intercept')
        not_significant = pvalues[pvalues > alpha]
        
        if not_significant.empty:
            break
        else:
            to_remove = not_significant.idxmax()
            exog_names.remove(to_remove)
            if i > 0:
                print()
            print(f"La variable explicative '{to_remove}' a été retirée du modèle",
                  f"(p-value : {pvalues[to_remove]:.2g}).")
            print(f"Variables restantes : {exog_names}")
            i += 1

    return res_wls


def check_collinearity(reg_result, threshold=5):
    exog = reg_result.model.exog
    title = "Variance inflation factor (VIF)"
    print(title)
    print("-" * len(title))
    vif_max = 0

    for i in range(1, exog.shape[1]):
        exog_name = reg_result.model.exog_names[i]
        vif = variance_inflation_factor(exog, i)
        print(f"{exog_name} : {vif:.1f}")
        vif_max = max(vif_max, vif)

    if vif_max < threshold:
        print(f"\nTous les VIF sont inférieurs à {threshold}, il n'y pas donc",
              "pas de problème de colinéarité.")


def logreg_backward_selection(X_train, y_train, add_const=False, alpha=0.05):
    if add_const:
        X_train = sm.add_constant(X_train)

    features = list(X_train.columns)

    if 'const' not in features:
        raise ValueError("Aucune constante (colonne `const`) trouvée. "
                         + "Utiliser add_const=True pour ajouter une constante.")

    while True:
        logreg = sm.Logit(endog=y_train, exog=X_train).fit()
        pvalues = logreg.pvalues.drop('const')
        not_significant = pvalues[pvalues > alpha]

        if not_significant.empty:
            break
        else:
            to_remove = not_significant.idxmax()
            features.remove(to_remove)
            print(f"\nLa variable explicative '{to_remove}' a été enlevée du modèle",
                  f"(p-value : {pvalues[to_remove]:.2g}).")
            print(f"Variables restantes : {features}")  # features.__repr__()[1:-1] pour enlever les crochets
            X_train = X_train[features]

    return logreg


def plot_elbow(X, k_max=10, x_vline=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    inertia = []
    k_array = np.arange(1, k_max + 1)
    
    for k in k_array:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    sns.lineplot(x=k_array, y=inertia, ax=ax, **kwargs)

    x_min, x_max = k_array.min(), k_array.max()
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(k_array)
    ax.set_xlabel("Nombre de classes (clusters)")
    ax.set_ylabel("Inertie")
    ax.set_title("Méthode du coude")

    if x_vline is not None:
        add_vline(x=x_vline, ax=ax)


def plot_silhouette_score(X, k_max=10, x_vline=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sil_scores = []
    k_array = np.arange(2, k_max + 1)
    
    for k in k_array:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        y = kmeans.fit_predict(X)
        sil_score = silhouette_score(X, y, metric='euclidean')
        sil_scores.append(sil_score)
    
    sns.lineplot(x=k_array, y=sil_scores, ax=ax, **kwargs)
    
    x_min, x_max = k_array.min(), k_array.max()
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(k_array)
    ax.set_xlabel("Nombre de clusters")
    ax.set_title("Coefficient de silhouette")

    if x_vline is not None:
        add_vline(x=x_vline, ax=ax)


def add_vline(x, ax=None, linewidth=3):
    if ax is None: ax = plt.gca()
    x_offset = 0
    
    # si la coordonnée x de la ligne verticale est égale à la limite inférieure
    # de l'axe des x, on décale la ligne d'une valeur égale à la moitié de
    # son épaisseur plus la moitié de l'épaisseur de l'axe des y
    if x == ax.get_xlim()[0]:
        y_axis_lw = ax.spines['left'].get_linewidth()
        x_offset_points = (linewidth / 2) + (y_axis_lw / 2)
        x_offset, _ = get_offset_in_data_units((x_offset_points, 0), ax)    
    
    ax.axvline(
        x=x + x_offset,
        color='red', linewidth=linewidth
    )


def get_offset_in_data_units(offset, ax):
    offset_data = (
        ax.transData.inverted().transform(offset)
        - ax.transData.inverted().transform((0, 0))
    )
    return offset_data


def get_cluster_mapper(kmeans_model, y):
    clusters = ['cluster_' + str(cluster) 
                for cluster in kmeans_model.labels_]
    crosstab = pd.crosstab(index=clusters, columns=y)
    mapper = {cluster: label for cluster, label
              in crosstab.idxmax(axis=1).items()}
    return mapper


def get_preds_kmeans(kmeans_model, X, mapper):
    clusters = kmeans_model.predict(X)
    preds = np.array(
        [mapper.get('cluster_' + str(cluster)) for cluster in clusters]
    )
    return preds


def center_text(text, width):
    q, r = divmod(width - len(text), 2)
    return q * ' ' + text + (q + r) * ' '


def format_confusion_matrix(conf_matrix):
    conf_matrix = pd.DataFrame(conf_matrix) \
        .rename_axis(index='Actual', columns='Predicted') \
        .rename(bool, axis=0) \
        .rename(bool, axis=1)
    return conf_matrix


@plt.rc_context({'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelsize': 14})
def evaluate_model(model, X_train, y_train, X_test, y_test, fitted=True):
    WIDTH = 80
    
    if not fitted:
        model.fit(X_train, y_train)
    
    # prédictions
    if isinstance(model, KMeans):
        mapper = get_cluster_mapper(model, y_train)
        y_train_pred = get_preds_kmeans(model, X_train, mapper)
        y_test_pred = get_preds_kmeans(model, X_test, mapper)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # courbe d'apprentissage
    if not isinstance(model, KMeans):
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_learning_curve(model, X, y, ax=ax)
        print()

    # train set
    print(center_text('Train set', WIDTH))
    print('=' * WIDTH)
    print()
    
    print('Matrice de confusion')
    print('--------------------')
    print(format_confusion_matrix(cm_train))
    
    print('\n')
    print('Rapport de classification')
    print('-------------------------')
    print(classification_report(y_train, y_train_pred))
    
    # test set
    print()
    print(center_text('Test set', WIDTH))
    print('=' * WIDTH)
    
    print()
    print('Matrice de confusion')
    print('--------------------')
    print(format_confusion_matrix(cm_test))
    
    
    print('\n')
    print('Rapport de classification')
    print('-------------------------')
    print(classification_report(y_test, y_test_pred))


def plot_learning_curve(model, X, y, ax=None):
    if ax is None:
        ax = plt.gca()
    
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1, 10),
        cv=5,
        scoring='f1'
    )
    
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    colors = sns.color_palette(n_colors=2)
    
    sns.lineplot(
        x=train_sizes,
        y=train_scores_mean,
        label='Training score',
        color=colors[0],
        ax=ax
    )
    
    ax.fill_between(
        x=train_sizes,
        y1=(train_scores_mean + train_scores_std),
        y2=(train_scores_mean - train_scores_std),
        color=colors[0],
        alpha=0.2
    )    
    
    sns.lineplot(
        x=train_sizes,
        y=val_scores_mean,
        label='Validation score',
        color=colors[1],
        ax=ax
    )
    
    ax.fill_between(
        x=train_sizes,
        y1=(val_scores_mean + val_scores_std),
        y2=(val_scores_mean - val_scores_std),
        color=colors[1],
        alpha=0.2
    )
    
    ax.set_title("Courbe d'apprentissage")
    ax.set_xlabel("Nombre d'observations dans le training set")
    ax.set_ylabel('Score F1')
    ax.set_xlim(train_sizes.min(), train_sizes.max())

    plt.show()


def scree_plot(pca, ax=None):
    n = pca.n_components_
    x_axis = range(1, n + 1)

    var_pct = pca.explained_variance_ratio_ * 100
    cum_var_pct = np.cumsum(var_pct)

    ax.bar(x_axis, var_pct)
    ax.plot(x_axis, cum_var_pct, color='red', marker='o')

    for x, y in zip(x_axis, cum_var_pct):
        ax.annotate(text=f'{y:.1f} %',
                    xy=(x, y), xycoords='data',
                    xytext=(0, 10), textcoords='offset points',
                    ha='center')

    ax.set_xlabel("Rang de l'axe d'inertie")
    ax.set_ylabel("Pourcentage d'inertie")
    ax.set_xticks(x_axis)
    ax.set_ylim(0, 115)
    ax.set_title("Eboulis des valeurs propres")


def get_text_alignment(x, y):   
    ha = 'right' if x <= 0 else 'left'
    va = 'top' if y <= 0 else 'bottom'
    return ha, va


def plot_correlation_circle(pca, components=(0, 1), features=None, ax=None):
    xcomp, ycomp = components
    
    for i in range(pca.components_.shape[1]):
        x = pca.components_[xcomp, i]
        y = pca.components_[ycomp, i]
        
        # texte
        ha, va = get_text_alignment(x, y)
        ax.annotate(
            features[i],
            xy=(x, y),
            ha=ha,
            va=va,
            bbox=dict(facecolor='lightgrey', alpha=0.5)
        )
        
        # flèches
        ax.annotate(
            '',
            xy=(0, 0),
            xytext=(x, y),
            arrowprops=dict(
                color=sns.color_palette()[0],
                lw=2,
                arrowstyle='<-'  # '<|-'
            )
        )
    
    # on représente un cercle de rayon 1
    angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(angles), np.sin(angles))
    ax.axis('equal')
    
    # on représente les axes
    ax.axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
    ax.axvline(x=0, color='grey', linestyle='dashed', linewidth=1)
    
    # nom des axes
    xcomp_var_pct = pca.explained_variance_ratio_[xcomp] * 100
    ycomp_var_pct = pca.explained_variance_ratio_[ycomp] * 100
    ax.set_xlabel('F{} ({:.1f} %)'.format(xcomp + 1, xcomp_var_pct))
    ax.set_ylabel('F{} ({:.1f} %)'.format(ycomp + 1, ycomp_var_pct))
    
    # titre                            
    ax.set_title('Cercle des corrélations (F{} et F{})'.format(xcomp + 1, ycomp + 1))


# la distance euclidienne a été ici calculée manuellement mais il existe la
# fonction np.linalg.norm qui permet de le faire plus simplement
def get_squared_euclidean_distance(X_scaled):
    """
    Calcule la distance euclidienne au carré des individus par rapport à
    l'origine/au centre de gravité du nuage des individus (en utilisant le
    théorème de Pythagore).
    """
    dist_sq = np.sum(X_scaled ** 2, axis=1, keepdims=True)
    return dist_sq


def get_cos_squared(X_scaled, X_projected):
    """
    Calcule le cosinus carré de l'angle entre Oi et OHi, où :
    - O représente l'origine du nuage ;
    - i représente un individu ;
    - Hi représente la projection de cet individu sur l'axe principal
      d'inertie.
    
    Plus cette valeur est proche de 1 et meilleure est la représentation de
    l'individu sur cet axe (plus la distance entre i et Hi est faible).
    
    cos^2 = OHi^2 / Oi^2
    """
    dist_sq = get_squared_euclidean_distance(X_scaled)
    cos_squared = X_projected ** 2 / dist_sq
    return cos_squared


def get_cos_squared_sum(X_scaled, X_projected, components=(0, 1)):
    cos_squared = get_cos_squared(X_scaled, X_projected)
    return np.sum(cos_squared[:, components], axis=1)


def plot_quality_of_projection(X_scaled, X_projected, y, components=(0, 1), ax=None):
    x_comp, y_comp = components
    if isinstance(y, pd.Series) and y.name is not None:
        label_name = y.name
    else:
        label_name = 'Etiquette'
    
    cos_squared_sum = get_cos_squared_sum(X_scaled, X_projected, components)
    
    data = pd.DataFrame({'x_comp': X_projected[:, x_comp],
                         'y_comp': X_projected[:, y_comp],
                         label_name: y,
                         'Cos carré': cos_squared_sum})
    
    sns.scatterplot(data=data, x='x_comp', y='y_comp',
                    hue=label_name,
                    size='Cos carré', sizes=(4, 48),
                    ax=ax)
    
    # on modifie les limites des axes des x et des y
    x_max = max(ax.get_xlim(), key=abs)
    y_max = max(ax.get_ylim(), key=abs)
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    
    ax.set_xlabel(f'F{x_comp + 1}')
    ax.set_ylabel(f'F{y_comp + 1}')
    
    ax.set_title(
        'Qualité de projection des points sur le\nplan factoriel '
        f'formé par F{x_comp + 1} et F{y_comp + 1}'
    )
    

@plt.rc_context({'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelsize': 14})
def plot_pca_caracteristics(pca, X_scaled, y_train, features=None):
    X_projected = pca.transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 16 / 3))
    
    scree_plot(pca, ax=axes[0])
    
    plot_correlation_circle(
        pca,
        components=(0, 1),
        features=features,
        ax=axes[1]
    )
    
    plot_quality_of_projection(
        X_scaled,
        X_projected,
        y_train,
        components=(0, 1),
        ax=axes[2]
    )
    
    fig.tight_layout()


@plt.rc_context({'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelsize': 14})
def plot_classification_using_pca_v2(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    resolution=1000,
    axes=None
):
    
    # on entraine l'ACP et on transforme les données
    pca = PCA(n_components=2)
    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)

    # pour chaque composante, on récupère les extrema auxquels on ajoute une
    # marge
    margin = np.ptp(X_train_transformed, axis=0) * 0.05
    f1_min, f2_min = X_train_transformed.min(axis=0) - margin
    f1_max, f2_max = X_train_transformed.max(axis=0) + margin

    # on crée une grille de points de taille `resolution` ** 2 dans le premier
    # plan factoriel de l'ACP
    f1_range = np.linspace(f1_min, f1_max, resolution)
    f2_range = np.linspace(f2_min, f2_max, resolution)
    xx, yy = np.meshgrid(f1_range, f2_range)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # on réalise la transformation inverse des points de cette grille
    grid_inverse_transformed = pca.inverse_transform(grid)

    # on prédit la valeur de y à la fois pour les points de la grille et
    # les points de X
    if isinstance(model, KMeans):
        mapper = get_cluster_mapper(model, y=y_train)
        grid_preds = get_preds_kmeans(model, grid_inverse_transformed,
                                      mapper=mapper)
        y_train_pred = get_preds_kmeans(model, X_train, mapper=mapper)
        y_test_pred = get_preds_kmeans(model, X_test, mapper=mapper)
    else:
        grid_preds = model.predict(grid_inverse_transformed)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # on représente les données
    colors = sns.color_palette(n_colors=2)
    palette = {label: color for label, color in enumerate(colors)}

    correctly_classified_train = pd.Series(data=(y_train == y_train_pred),
                                           name='is_correctly_classified')
    
    correctly_classified_test = pd.Series(data=(y_test == y_test_pred),
                                          name='is_correctly_classified')
    
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        ax.contourf(xx, yy, grid_preds.reshape(xx.shape),
                    levels=[0, 0.5, 1], colors=colors, alpha=0.2)
    
    # train set
    sns.scatterplot(
        x=X_train_transformed[:, 0], y=X_train_transformed[:, 1],
        hue=y_train, palette=palette,
        size=correctly_classified_train, sizes=[25, 64], size_order=[True, False],
        style=correctly_classified_train, markers=['o', 'X'], style_order=[True, False],
        ax=axes[0]
    )

    # test set
    sns.scatterplot(
        x=X_test_transformed[:, 0], y=X_test_transformed[:, 1],
        hue=y_test, palette=palette,
        size=correctly_classified_test, sizes=[25, 64], size_order=[True, False],
        style=correctly_classified_test, markers=['o', 'X'], style_order=[True, False],
        ax=axes[1]
    )
    
    # pour les kmeans, on représente également la position des centroïdes
    if isinstance(model, KMeans):
        centers_transformed = pca.transform(model.cluster_centers_)
        
        for ax in axes:
            sns.scatterplot(
                x=centers_transformed[:, 0], y=centers_transformed[:, 1],
                color='black', marker='s',
                ax=ax
            )

            for i in range(model.n_clusters):
                cluster_pred = mapper.get('cluster_' + str(i))
                text = f'cluster {i}\n(y = {cluster_pred})'
                xy = centers_transformed[i, :]
                ax.annotate(
                    text, xy,
                    xytext=(0, 10), textcoords='offset pixels',
                    ha='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)
                )

    # on met en forme les axes
    f1_var_pct = pca.explained_variance_ratio_[0] * 100
    f2_var_pct = pca.explained_variance_ratio_[1] * 100
    sum_var_pct = f1_var_pct + f2_var_pct
    
    set_names = ['train set', 'test set']
    
    for i, ax in enumerate(axes):
        ax.set_xlabel(f'F1 ({f1_var_pct:.1f} %)')
        ax.set_ylabel(f'F2 ({f2_var_pct:.1f} %)')
        ax.set_title(
            f'Projection des observations du {set_names[i]} \nsur les axes '
            + f'F1 et F2 ({sum_var_pct:.1f} %)'
        )

    axes[0].get_legend().remove()
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_n_best_scores(cv_results, n=3, y_bottom_lim=None):  
    n_best = pd.DataFrame(cv_results) \
        .nsmallest(n, columns='rank_test_score')
    
    cols = [col for col in n_best.columns if col.startswith('split')]
    
    n_best_melted = n_best \
        .reset_index() \
        .melt(id_vars='index', value_vars=cols)
    n_best_melted['index'] = n_best_melted['index'].astype(str)
    
    fig, ax = plt.subplots()
    sns.barplot(data=n_best_melted, x='index', y='value', ax=ax)
    
    # limites de l'axe des y
    if y_bottom_lim is not None:
        # y_max = max(line.get_ydata()[1] for line in ax.lines)
        y_max = ax.get_ylim()[1] / 1.05
        top_lim = y_max + (y_max - y_bottom_lim) * 0.05
        ax.set_ylim(y_bottom_lim, top_lim)
    
    ax.set_ylabel('score')
    display(fig)
    plt.close(fig)
    
    for index, params in n_best['params'].items():
        print(f'{index}:\n{params}')