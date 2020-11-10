import click

from src.main import predict as predict_f
from src.utils.model import load_h5_model, train_model, evaluate_model,
    smile_to_pred


@click.group("servier")
def cli(*args, **kwargs):
    pass

@cli.command()
@click.argument("smile")
@click.argument("model_path", required=False, default="models/cnn_1d_1110.h5")
def predict(smile, model_path):
    model = load_h5_model(model_path)
    smile_to_pred(smile, model)

@cli.command()
@click.argument("data_dir", required=False, default="data")
def train(data_dir):
    train_model(data_dir=data_dir)

@cli.command()
@click.argument("data_dir", required=False, default="data")
@click.argument("model_path", required=False, default="models/cnn_1d_1110.h5")
def evaluate(data_dir, model_path):
    evaluate_model(data_dir=data_dir, model_path=model_path)
