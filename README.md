## PyANN - Generic Artificial Neural Network Framework in Python

This is a general framework for creating neural nets, through the use of script files and case files.
Script files define the neural net you want to create, case files contain optional training and testing data.

The framework was written as part of a series of exercises in the course _IT3708 Subsymbolic Methods in AI_ at [NTNU](http://ntnu.no).
For more information, see our [expercise report](PyANN/blob/master/report.pdf?raw=true) describing the framework implementation and how we used it to train a ANN [webots](http://www.cyberbotics.com/overview) controller.

We haven't bothered to write any user documentation, so take instead a look at the examples in the `annlib/scripts` and `annlib/cases` folders.

The framework's only dependency should be the python `yaml` package.
