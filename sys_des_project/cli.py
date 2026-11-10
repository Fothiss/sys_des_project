"""CLI конвейера обработки."""

import typer

from sys_des_project.pipeline import convert, s3_download, split, stats, upload

app = typer.Typer()

app.command("download")(s3_download.main)
app.command("convert")(convert.main)
app.command("split")(split.main)
app.command("stats")(stats.main)
app.command("upload")(upload.main)

if __name__ == "__main__":
    app()
