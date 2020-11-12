import click

from src.utils.model import load_models, train_model, evaluate_model, smile_to_pred


@click.group("servier")
@click.version_option("0.0.1")
def cli(*args, **kwargs):
    """servier - Property molecule prediction and model management"""
    pass

@cli.command()
@click.argument("smile")
@click.option("--model_1_path", default="../models/model_1.h5", help="Path of the model 1 checkpoint")
@click.option("--model_2_path", default="../models/model_2.h5", help="Path of the model 2 checkpoint")
@click.option("--model_name", default="2", help="Model to be used for retrieving prediction")
def predict(smile, model_1_path, model_2_path, model_name):
    """
    Predict P1 property based on molecule fingerprint.
    """
    model_1, model_2 = load_models(model_1_path, model_2_path)
    smile_to_pred(smile, model_1, model_2, model_name)

@cli.command()
@click.option("--data_dir", default="../data", help="Path of the data directory")
@click.option("--model_name", default="2", help="Model to be trained")
def train(data_dir, model_name):
    """
    Train model from scratch.
    """
    train_model(data_dir=data_dir, model_name=model_name)

@cli.command()
@click.option("--data_dir", default="../data", help="Path of the data directory")
@click.option("--model_path", default="../models/model_2.h5", help="Path of the model to be evaluated")
@click.option("--model_name", default="2", help="Model to be evaluated")
def evaluate(data_dir, model_path, model_name):
    """
    Evaluate model.
    """
    evaluate_model(data_dir=data_dir, model_path=model_path, model_name=model_name)
