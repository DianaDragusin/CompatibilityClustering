import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request
from kmeans import KMeansClustering

app = Flask(__name__)

kmeans = KMeansClustering(number_of_clusters=5)


def custom_scale(data, feature_range):
    a, b = feature_range

    col_min = 1
    col_max = 5
    scaled_data = a + ((data - col_min) * (b - a) / (col_max - col_min))

    return scaled_data


def custom_invert_scale(data, feature_range=(-1, 1)):
    col_min = 1
    col_max = 5
    col_mid = (col_max + col_min) / 2

    # Inverting the column values dynamically around the midpoint
    data = col_mid - (data - col_mid)

    # Scaling to the specified feature range
    a, b = feature_range
    scaled_data = a + ((data - col_min) * (b - a) / (col_max - col_min))

    return scaled_data


def train_model():
    filename = "bigFiveDataset/data-final.csv"
    read_data = pd.read_csv(filename, delimiter='\t')
    read_data = read_data.iloc[:, :50]

    read_data = read_data.replace(0, np.nan)
    read_data.fillna(read_data.median(), inplace=True)

    ext_questions = {'EXT1': 'I am the life of the party.',
                     'EXT2': 'I don\'t talk a lot.',
                     'EXT3': 'I feel comfortable around people.',
                     'EXT4': 'I keep in the background.',
                     'EXT5': 'I start conversations.',
                     'EXT6': 'I have little to say.',
                     'EXT7': 'I talk to a lot of different people at parties.',
                     'EXT8': 'I don\'t like to draw attention to myself.',
                     'EXT9': 'I don\'t mind being the center of attention.',
                     'EXT10': 'I am quiet around strangers.'}

    neu_questions = {'EST1': 'I get stressed out easily.',  # nervous
                     'EST2': 'I am relaxed most of the time.',  # confident
                     'EST3': 'I worry about things.',  # nervous
                     'EST4': 'I seldom feel blue.',  # confident
                     'EST5': 'I am easily disturbed.',  # nervous
                     'EST6': 'I get upset easily.',  # nervous
                     'EST7': 'I change my mood a lot.',  # nervous
                     'EST8': 'I have frequent mood swings.',  # nervous
                     'EST9': 'I get irritated easily',  # nervous
                     'EST10': 'I often feel blue.'}  # nervous

    # measure of one's trusting and helpful nature
    agr_questions = {'AGR1': 'I feel little concern for others.',  # detached
                     'AGR2': 'I am interested in people.',  # friendly
                     'AGR3': 'I insult people.',  # detached
                     'AGR4': 'I sympathize with others feelings.',  # friendly
                     'AGR5': 'I am not interested in other people problems.',  # detached
                     'AGR6': 'I have a soft heart',  # friendly
                     'AGR7': 'I am not really interested in others.',  # detached
                     'AGR8': 'I take time out for others',  # friendly
                     'AGR9': 'I feel others emotions',  # friendly
                     'AGR10': 'I make people feel at ease.'}  # friendly

    # tendency to be organized and dependable
    con_questions = {'CSN1': 'I am always prepared.',  # organized
                     'CSN2': 'I leave my belongings around.',  # careless
                     'CSN3': 'I pay attention to details',  # organized
                     'CSN4': 'I make a mess of things.',  # careless
                     'CSN5': 'I get chores done right away.',  # organized
                     'CSN6': 'I often forget to put things back in their proper place.',  # careless
                     'CSN7': 'I like order.',  # organized
                     'CSN8': 'I shirk my duties.',  # careless
                     'CSN9': 'I follow a schedule.',  # organized
                     'CSN10': 'I am exacting in my work.'}  # organized

    # degree of intellectual curiosity, creativity and a preference for novelty
    opn_questions = {'OPN1': 'I have a rich vocabulary.',  # inventive
                     'OPN2': 'I have difficulty understanding abstract ideas.',  # cautious/consistent
                     'OPN3': 'I have a vivid imagination.',  # inventive
                     'OPN4': 'I am not interested in abstract ideas.',  # cautious/consistent
                     'OPN5': 'I have excellent ideas',  # inventive
                     'OPN6': 'I do not have a good imagination.',  # cautious/consistent
                     'OPN7': 'I am quick to understand things.',  # inventive
                     'OPN8': 'I use difficult words.',  # inventive
                     'OPN9': 'I spend time reflecting on things.',  # inventive
                     'OPN10': 'I am full of ideas.'}  # inventive

    changed_qs = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
                  'EST2', 'EST4',
                  'AGR1', 'AGR3', 'AGR5', 'AGR7',
                  'CSN2', 'CSN4', 'CSN6', 'CSN8',
                  'OPN2', 'OPN4', 'OPN6']

    for column in read_data:
        if column in changed_qs:
            read_data[column] = custom_invert_scale(read_data[column], feature_range=(-1, 1))
        else:
            read_data[column] = custom_scale(read_data[column], feature_range=(-1, 1))

    def summation(ocean_dict, label):
        read_data[label] = 0
        for question in ocean_dict.keys():
            read_data[label] += read_data[question]

    summation(ext_questions, 'extroversion_score')
    summation(neu_questions, 'neuroticism_score')
    summation(agr_questions, 'agreeableness_score')
    summation(con_questions, 'conscientiousness_score')
    summation(opn_questions, 'openness_score')

    data_for_training = read_data[['extroversion_score', 'neuroticism_score', 'agreeableness_score',
                                   'conscientiousness_score', 'openness_score']].copy()

    kmeans.fit(data_for_training.to_numpy())

    predictions = kmeans.predict(data_for_training.to_numpy())
    data_for_training['clusters'] = predictions

    desc = data_for_training.groupby('clusters')[['extroversion_score', 'neuroticism_score', 'agreeableness_score',
                                                  'conscientiousness_score', 'openness_score']].describe()
    summary = pd.concat(
        objs=(i.set_index('clusters') for i in (
            desc['extroversion_score'][['count', 'mean']].reset_index(),
            desc['neuroticism_score'][['mean']].reset_index(),
            desc['agreeableness_score'][['mean']].reset_index(),
            desc['conscientiousness_score'][['mean']].reset_index(),
            desc['openness_score'][['mean']].reset_index())),
        axis=1,
        join='inner').reset_index()
    summary.columns = ['clusters', 'cluster_count', 'extroversion_mean', 'neuroticism_mean', 'agreeableness_mean',
                       'conscientiousness_mean', 'openness_mean']

    plt.figure(figsize=(9, 7))
    summary.plot(x='clusters',
                 y=['extroversion_mean', 'neuroticism_mean', 'agreeableness_mean', 'conscientiousness_mean',
                    'openness_mean'],
                 kind='bar',
                 ylabel='Mean of characteristics',
                 fontsize=14, cmap="tab20b").legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                                    labels=['Mean of Extroversion',
                                                            'Mean of Neuroticism',
                                                            'Mean of Agreeableness',
                                                            'Mean of Conscientiousness',
                                                            'Mean of Openness'])
    plt.title('The 5 Clusters of BIG5 Personality Test', fontsize=14)
    plt.show()


@app.route("/personality-cluster-prediction", methods=['POST'])
def personality_cluster_prediction():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data]).to_numpy()

    prediction = kmeans.predict(input_df)

    prediction = int(prediction[0])

    return jsonify(prediction)


if __name__ == '__main__':
    with app.app_context():
        train_model()
        app.run()