## RRICT Assist (Placeholder name for now)
RRICT Assist is an AI computer assistant

## Project Structure
The project will be composed of the following modules:

- Datasets
- Action classifier
- Action Handler
- Entity extractor

### Datasets
Datasets will be stored in the `datasets/` directory

### Classifier
The Action classifier model and source will be stored in
`src/action_classifier`. After training the training parameters will be stored
in `trained/<model name>-<trained date & time>.format`.

### Handler
The source code for the Action Handler will be stored in `src/action_handler`

## TODO:
- Gather training data
- Train classifier
- Write action handler
