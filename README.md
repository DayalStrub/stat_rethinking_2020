Statistical Rethinking: A Bayesian Course (with Code Examples in R/Stan/Python/Julia)
===============

Forked.

# Topical Outline

| Week ## | Meeting date | Reading | Lectures | Numpyro | Turing | 
| ------- | -------------- | ------------- | ---------------------- | --------------- | ------------- | 
| Week 01 | 02 December  | Chapters 1, 2 and 3 | The Golem of Prague <[slides](https://speakerdeck.com/rmcelreath/l01-statistical-rethinking-winter-2019)> <[video](https://www.youtube.com/watch?v=4WVelCswXo4)> <br>Garden of Forking Data <[slides](https://speakerdeck.com/rmcelreath/l02-statistical-rethinking-winter-2019)> <[video](https://www.youtube.com/watch?v=XoVtOAN0htU&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI&index=2)> |
| Week 02 | 09 December | Chapter 4 | Geocentric Models <[slides](https://speakerdeck.com/rmcelreath/l03-statistical-rethinking-winter-2019)> <[video](https://youtu.be/h5aPo5wXN8E)><br> Wiggly Orbits <[slides](https://speakerdeck.com/rmcelreath/l04-statistical-rethinking-winter-2019)> <[video](https://youtu.be/ENxTrFf9a7c)>  
| Week 03 | 06 January | Chapters 5 and 6 | Spurious Waffles <[slides](https://speakerdeck.com/rmcelreath/l05-statistical-rethinking-winter-2019)> <[video](https://www.youtube.com/watch?v=e0tO64mtYMU&index=5&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI)> <br>Haunted DAG <[slides](https://speakerdeck.com/rmcelreath/l06-statistical-rethinking-winter-2019)> <[video](https://youtu.be/l_7yIUqWBmE)> 
| Week 04 | 13 January | Chapter 7 | Ulysses' Compass <[slides](https://speakerdeck.com/rmcelreath/l07-statistical-rethinking-winter-2019)> <[video](https://youtu.be/0Jc6Kgw5qc0)>  <br>Model Comparison <[slides](https://speakerdeck.com/rmcelreath/l08-statistical-rethinking-winter-2019)> <[video](https://youtu.be/gjrsYDJbRh0)> 
| Week 05 | 20 January | Chapters 8 and 9 | Conditional Manatees <[slides](https://speakerdeck.com/rmcelreath/l09-statistical-rethinking-winter-2019)> <[video](https://youtu.be/QhHfo6-Bx8o)> <br>Markov Chain Monte Carlo <[slides](https://speakerdeck.com/rmcelreath/l10-statistical-rethinking-winter-2019)> <[video](https://youtu.be/v-j0UmWf3Us)>  
| Week 06 | 27 January | Chapters 10 and 11 | Maximum entropy & GLMs <[slides](https://speakerdeck.com/rmcelreath/l11-statistical-rethinking-winter-2019)> <[video](https://youtu.be/-4y4X8ELcEM)> <br>God Spiked the Integers <[slides](https://speakerdeck.com/rmcelreath/l12-statistical-rethinking-winter-2019)> <[video](https://youtu.be/hRJtKCIDTwc)>
| Week 07 | 03 February | Chapter 12 | Monsters & Mixtures <[slides](https://speakerdeck.com/rmcelreath/l13-statistical-rethinking-winter-2019)> <[video](https://youtu.be/p7g-CgGCS34)> <br>Ordered Categories, Left & Right <[slides](https://speakerdeck.com/rmcelreath/l14-statistical-rethinking-winter-2019)> <[video](https://youtu.be/zA3Jxv8LOrA)> 
| Week 08 | 10 February | Chapter 13 | Multilevel Models <[slides](https://speakerdeck.com/rmcelreath/l15-statistical-rethinking-winter-2019)> <[video](https://youtu.be/AALYPv5xSos)> <br>Multilevel Models 2 <[slides](https://speakerdeck.com/rmcelreath/l16-statistical-rethinking-winter-2019)> <[video](https://youtu.be/ZG3Oe35R5sY)>  
| Week 09 | 24 February | Chapter 14 | Adventures in Covariance <[slides](https://speakerdeck.com/rmcelreath/l17-statistical-rethinking-winter-2019)> <[video](https://youtu.be/yfXpjmWgyXU)> <br> Slopes, Instruments and Social Relations <[slides](https://speakerdeck.com/rmcelreath/l18-statistical-rethinking-winter-2019)> <[video](https://youtu.be/e5cgiAGBKzI)>
| Week 10 | 03 March | Chapter 15 | Gaussian Processes <[slides](https://speakerdeck.com/rmcelreath/l19-statistical-rethinking-winter-2019)> <[video](https://youtu.be/pwMRbt2CbSU)>  <br>Missing Values and Measurement Error <[slides](https://speakerdeck.com/rmcelreath/l20-statistical-rethinking-winter-2019)> <[video](https://youtu.be/UgLF0aLk85s)>  

# Code examples

## R 

### Original R Flavor

For those who want to use the original R code examples in the print book, you need to first install `rstan`. Go to <http://mc-stan.org/> and find the instructions for your platform. Then you can install the `rethinking` package:
```
install.packages(c("devtools","mvtnorm","loo","coda"),dependencies=TRUE)
library(devtools)
install_github("rmcelreath/rethinking")
```
The code is all on github <https://github.com/rmcelreath/rethinking/> and there are additional details about the package there, including information about using the more-up-to-date `cmdstanr` instead of `rstan` as the underlying MCMC engine.

### R + Tidyverse + ggplot2 + brms

The <[Tidyverse/brms](https://bookdown.org/content/4857/)> conversion is very high quality and complete through Chapter 14.

## Python 

### Numpyro

[Book](https://fehiepsi.github.io/rethinking-numpyro/) and [course](https://github.com/asuagar/statrethink-course-numpyro-2019). See also numpyro documentation.

### PyMC3

The <[Python/PyMC3](https://github.com/pymc-devs/resources/tree/master/Rethinking_2)> conversion is quite complete.

## Julia: Turing

The <[Julia/Turing](https://github.com/StatisticalRethinkingJulia)> conversion is not as complete, but is growing fast and presents the Rethinking examples in multiple Julia engines, including the great <[TuringLang](https://github.com/StatisticalRethinkingJulia/TuringModels.jl)>.

Another Turing [port](https://shmuma.github.io/rethinking-2ed-julia/).

## Other

The are several other conversions. See the full list at <https://xcelab.net/rm/statistical-rethinking/>.

# Environment(s)

```bash
sudo docker build --file Dockerfile --tag sr .
sudo docker run -p 8888:8888 -v "$PWD":/home/jovyan/notebooks sr jupyter lab --ContentsManager.allow_hidden=True --allow-root
```


