def save_model(model, directory='./models'):
    """
    creates a file name by appending .pickle' to a model's variable name,
    and saves the model as a pickle file in the working directory's
    'model' subdirectory
    """
    # verify or create the save - path directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    # build the file name
    f_name = [tuple[0] for tuple in filter(
        lambda x: model is x[1],
        globals().items())
             ][0]

    # pickle the model with the created filename
    with open(f'{directory}/{f_name}.pickle', 'wb') as f:
        # pickling the dataframe using the highest protocol available
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    return print(f'Saved to {directory} as pickle file: {f_name}\n', '-'*50)


def save_models(models):
    """
    passes a list of model names and saves
    the models as separate pickle files
    """
    for m in models:
        saved_model = save_model(m)
    return saved_model


def model_name(model):
    """
    returns a model variable as a string
    """
    return [tuple[0] for tuple in filter(
        lambda x: model is x[1],
        globals().items())
             ][0]
