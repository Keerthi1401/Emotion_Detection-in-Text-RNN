rm(list = ls())

# Set working directory
my_wd <- "D:\\UNCC\\EmoDetct"
setwd(paste(my_wd, "\\novels_cleaned_wasserstein_distance", sep = ''))

# Install packages
for (lib in c('weights',
              'data.table',
              'stats',
              'Hmisc',
              'TDA',
              'randomForest',
              'caret')) {
  if (!lib %in% installed.packages()[, 1]) {
    install.packages(lib)
  }
}

rm(lib)
library(weights)
library(data.table)
library(stats)
library(Hmisc)
library(TDA)
library(caret)

files <- list.files(path = ".")
files <- files[grepl('t_zero', files)]
out.files <- gsub('.t_zero.csv', '.png', files)

for (i in 1:length(files)) {
  setwd(paste(my_wd, "\\novels_cleaned_wasserstein_distance", sep = ''))
  novel <- read.csv(files[i])
  novel <- novel / mean(colMeans(novel))
  maxscale = 1 # limit of the filtration
  maxdimension = 2 # components and loops
  Diag <- ripsDiag(
    X = novel,
    maxdimension,
    maxscale,
    dist = "arbitrary",
    library = "Dionysus",
    printProgress = FALSE
  )$diagram
  setwd(paste(my_wd, "\\novels_diagrams", sep = ''))
  png(
    filename = out.files[i],
    width = 4000,
    height = 3000,
    res = 1000
  )
  plot(Diag, main = "Rips Diagram")
  dev.off()
}

files.select <- files[grepl('Devils', files) |
                        grepl('Idiot', files) |
                        grepl('Karamazov', files) |
                        grepl('Mansfield', files) |
                        grepl('Emma', files) |
                        grepl('Sense and Sensibility', files) |
                        grepl('Kenilworth', files) |
                        grepl('Guy Mannering', files) |
                        grepl('Rob Roy', files)]
out.files.select <- gsub('.t_zero.csv', '', files.select)
out.files.select <- gsub('\\(The Devils\\)', '', out.files.select)
out.files.select <- gsub(' - Complete', '', out.files.select)
out.files.select <- gsub(' - Complete', '\n', out.files.select)
out.files.select <- gsub('\\.', '\n', out.files.select)

setwd(paste(my_wd, "\\novels_diagrams", sep = ''))
png(
  filename = '3by3.9.diagrams.png',
  width = 16000,
  height = 10000,
  res = 1000
)
par(mfrow = c(3, 3))

for (i in 1:length(files.select)) {
  setwd(paste(my_wd, "\\novels_cleaned_wasserstein_distance", sep = ''))
  novel <- read.csv(files.select[i])
  novel <- novel / mean(colMeans(novel))
  maxscale = 1 # limit of the filtration
  maxdimension = 3 # components and loops
  Diag <-
    ripsDiag(
      X = novel,
      maxdimension,
      maxscale,
      dist = "arbitrary",
      library = "Dionysus",
      printProgress = FALSE
    )$diagram
  plot(
    Diag,
    main = out.files.select[i],
    cex.main = 2.7,
    cex.lab = 2.2,
    cex.axis = 3.5
  )
  legend(
    0.62,
    0.4,
    c("Component", "Loop"),
    col = c(1, 2),
    pch = c(20, 2),
    cex = 1.75,
    pt.lwd = 2
  )
}
dev.off()

# Save list of novels
novels <- as.data.frame(matrix(0, ncol = 2, nrow = 75))
names(novels) <- c("Novel", "Writer")
for (i in 1:75) {
  novels$Writer[i] <- strsplit(files[i], split = '\\.')[[1]][1]
  novels$Novel[i] <- strsplit(files[i], split = '\\.')[[1]][2]
}

setwd(my_wd)
fwrite(novels, "List.of.Novels.csv")
