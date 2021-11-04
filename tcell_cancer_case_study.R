library(tidyr)

##### MaxQuant Stats ###########
data <- read.delim('ion_mobility/PXD023049/txt/evidence.txt', stringsAsFactors = FALSE)
data <- data[grep("CON__|REV__", data$Leading.razor.protein, invert=TRUE),]
data <- data[data$Charge > 1,]
data$PeptideID <- paste(data$Modified.sequence, data$Charge , sep = "_")
y_dda <- aggregate(Intensity ~ Raw.file + PeptideID, 
                   FUN = function(x) max(x, na.rm=TRUE),
                   na.action = na.pass, data = data)




y <- spread(y_dda, key = Raw.file, value = Intensity)

rownames(y) <- y[,1]
y <- y[,-1]

# proportion of peptides with complete measurements
mean(complete.cases(y))


# number of partially observed peptides
sum(!complete.cases(y))




# proportion of missing values
mean(!is.na(y))


# missing completely or partially in cancer samples
sum(!complete.cases(y[,1:3]) & complete.cases(y[,4:6]))
missing_peptides <- rownames(y)[!complete.cases(y[,1:3]) & complete.cases(y[,4:6])]

y_missing_in_cancer <- y[!complete.cases(y[,1:3]) & complete.cases(y[,4:6]),]
dim(y_missing_in_cancer)



## and top N such peptides
head(y[!complete.cases(y[,1:3]) & complete.cases(y[,4:6]),], n= 30)


## depth of protein coverage
length(unique(data$Leading.razor.protein))



#### Label propagation Stats ############
# allPeptidestxt <- read.delim('ion_mobility/PXD023049/txt/allPeptides.txt',
#                              stringsAsFactors = FALSE)
# 
# 
# allPeptidestxt[allPeptidestxt$Charge==2 & 
#                  allPeptidestxt$Raw.file == 'Control_secretoma1_Slot2-19_1_2497' &
#                  allPeptidestxt$Ion.mobility.index == 552 &
#                  allPeptidestxt$Retention.time == round(54.028,1),]



reference_data = read.csv('ion_mobility/reference_embeddings_and_metadata_icml2021.csv')
query_data = read.csv('ion_mobility/query_embeddings_and_metadata_icml2021.csv')


library(FNN)


################
# final approach
###############

get.prototypes <- function(latent, labels, prototype_labels){
  # sapply(prototype_labels, FUN=function(x) colMeans(latent[labels == x,]))
  prototypes <- matrix(0, nrow = ncol(latent), ncol = length(prototype_labels))
  for (i in seq_along(prototype_labels)) {
    if (i%%1000 == 0)  message(i)
    prototypes[,i] <- colMeans(latent[labels == prototype_labels[i],])
  }
  
  return(prototypes)
}

embedding <- reference_data[,grep('dim', colnames(reference_data))]
labels <- reference_data$labels

length(missing_peptides)
sum(missing_peptides %in% labels)

missing_peptides <- missing_peptides[missing_peptides %in% labels]


prototypes <- get.prototypes(embedding, labels, missing_peptides)
colnames(prototypes) <- missing_peptides
rownames(prototypes) <- paste0("dim_",0:9)

query_embedding <- query_data[,#query_data$Isotope.correlation > 0.5 , # choose 0.9860 for top 25% high abundance 
                              grep('dim', colnames(query_data))]



dim(query_embedding)

knn_prototypes <- get.knnx(t(prototypes), query_embedding, k = 1)
# probs <- exp(-0.5*((knn_prototypes$nn.dist^2)/sd(knn_prototypes$nn.dist)))
probs <- exp(-0.5*((knn_prototypes$nn.dist^2)))
# probs <- probs/sum(probs)

summary(probs)


idxs <- apply(probs, 1, which.max) # this is always the first nearest neighbor
table(idxs)
max_probs <- probs[,1]
query_idents <- colnames(prototypes)[knn_prototypes$nn.index[,1]]

df_query_idents <- cbind(
                         query_data[,#query_data$Isotope.correlation > 0.5,
                                    grep('dim|X', colnames(query_data), invert = TRUE)],
                         data.frame(p = max_probs, PeptideID = query_idents))




table(df_query_idents$p > 0.8)
table(df_query_idents$p > 0.5)
length(unique(df_query_idents$PeptideID))


transfered_idents <- df_query_idents[df_query_idents$p >= 0.5,]

# transfers should exclude features currently identified in the runs
# transfered_idents$ident <- paste(transfered_idents$PeptideID)


y_new <- rbind(y_dda, 
               transfered_idents[ ,c("Raw.file","PeptideID","Intensity")]
               )
y_new <- aggregate(Intensity ~ Raw.file + PeptideID, 
                   FUN = function(x) max(x, na.rm=TRUE),
                   na.action = na.pass, data = y_new)



y_new <- spread(y_new, key = Raw.file, value = Intensity)

rownames(y_new) <- y_new[,1]
y_new <- y_new[,-1]


mean(!is.na(y_new))
mean(!is.na(y))

mean(!is.na(y_missing_in_cancer))
mean(!is.na(y_new[missing_peptides,]))

ynew_missing_in_cancer <- y_new[missing_peptides,]
ynew_missing_in_cancer[!is.na(y_missing_in_cancer)] <- y_missing_in_cancer[!is.na(y_missing_in_cancer)]


round(mean(!is.na(y_missing_in_cancer)),2)
round(mean(!is.na(ynew_missing_in_cancer)),2)

########################################################
## another approach is to label propagation per run:
#######################################################


# screen all prototypes first

missing_peptides <- rownames(y_missing_in_cancer)
missing_peptides <- missing_peptides[missing_peptides %in% labels]


prototypes <- get.prototypes(embedding, labels, missing_peptides)
colnames(prototypes) <- missing_peptides
rownames(prototypes) <- paste0("dim_",0:9)




transfered_idents <- list()
for (run_id in colnames(y_missing_in_cancer)[1:3]){
  message(run_id)
  missing_idents <- rownames(y_missing_in_cancer)[is.na(y_missing_in_cancer[,run_id])]
  message("Number of missing idents")
  message(length(missing_idents))
   # use pre-computed prototypes
  run_prototypes <- prototypes[, colnames(prototypes) %in% missing_idents]
  prototype_charges <- as.numeric(gsub("(.*)__(.*)","\\2", colnames(prototypes)))
  
  query_embedding <- query_data[query_data$Raw.file %in% run_id,#query_data$Isotope.correlation > 0.5 , # choose 0.9860 for top 25% high abundance 
                                grep('dim', colnames(query_data))]
  
  query_charge <- query_data$Charge[query_data$Raw.file %in% run_id]
  knn_prototypes <- get.knnx(t(run_prototypes), query_embedding, k = 5)
  
  
  
  probs <- exp(-0.5*((knn_prototypes$nn.dist^2)/matrixStats::rowSds(knn_prototypes$nn.dist)))
  
  
  
  ww <- matrix(prototype_charges[knn_prototypes$nn.index], nrow = nrow(probs), ncol = ncol(probs))
  charge <- matrix(query_charge, nrow = nrow(ww), ncol = ncol(ww), byrow = FALSE)
  w <- ifelse(ww==charge, 1, 0)
  
  wprobs <- w*probs
  
  p1 <- wprobs
  p2 <- wprobs/rowSums(probs)
  p3 <- wprobs/rowSums(wprobs)
  
  # # probs <- wprobs/sum(probs)
  table(p1 > 0.5)
  table(p2 > 0.5)
  table(p3 > 0.5)
  
  
  idxs <- apply(p3, 1, FUN= function(x) x == max(x))
  
  valid_features <- complete.cases(t(idxs))
  
  t(p3[valid_features,])[idxs[, valid_features]] #works but needs nan queries removed
  
  max_probs <- t(p3[valid_features,])[idxs[, valid_features]]
  
  ident_max_probs <- t(knn_prototypes$nn.index[valid_features,])[idxs[, valid_features]]
  query_idents <- colnames(prototypes)[ident_max_probs]
  
  
  df_query_idents <- cbind(
    query_data[query_data$Raw.file %in% run_id,#query_data$Isotope.correlation > 0.5,
               grep('dim|X', colnames(query_data), invert = TRUE)][valid_features,],
    data.frame(p = max_probs, PeptideID = query_idents))
  
  transfered_idents[[run_id]] <- df_query_idents
  
}

transfered_idents <- do.call(rbind, transfered_idents)
# transfered_idents$charges_agree <- ifelse(as.numeric(gsub("(.*)__(.*)","\\2", transfered_idents$PeptideID)) == transfered_idents$Charge,
#                                           TRUE, FALSE)

table(transfered_idents$p > 0.8 & transfered_idents$charges_agree)
table(transfered_idents$p > 0.5 & transfered_idents$charges_agree)



y_new <- rbind(y_dda, 
               transfered_idents[transfered_idents$p > 0.5  ,
                                 c("Raw.file","PeptideID","Intensity")]
)
y_new <- aggregate(Intensity ~ Raw.file + PeptideID, 
                   FUN = function(x) max(x, na.rm=TRUE),
                   na.action = na.pass, data = y_new)



y_new <- spread(y_new, key = Raw.file, value = Intensity)

rownames(y_new) <- y_new[,1]
y_new <- y_new[,-1]


mean(!is.na(y_new))
mean(!is.na(y))



ynew_missing_in_cancer <- y_new[missing_peptides,]
#ynew_missing_in_cancer[!is.na(y_missing_in_cancer)] <- y_missing_in_cancer[!is.na(y_missing_in_cancer)]


round(mean(!is.na(y_missing_in_cancer)),2)
round(mean(!is.na(ynew_missing_in_cancer)),2)


library(pheatmap)

z1 <- t(scale(t(log2(y_missing_in_cancer))))
z2 <- t(scale(t(log2(ynew_missing_in_cancer))))

pheatmap(z1, show_rownames = FALSE,treeheight_row = 0, treeheight_col = 0, labels_col = gsub("(.*)_(.*)_(.*)_(.*)_(.*)", "\\1_\\2", colnames(z1)), main = "MaxQuant")
pheatmap(z2, show_rownames = FALSE, treeheight_row = 0, treeheight_col = 0, labels_col = gsub("(.*)_(.*)_(.*)_(.*)_(.*)", "\\1_\\2", colnames(z2)), main = "PIP peptideprotonet")




















##################
# first approach
##################

get.prototypes <- function(latent, labels){
  sapply(unique(labels), FUN=function(x) colMeans(latent[labels == x,]))
  
}

prototypes <- get.prototypes(reference_data[,grep('dim', colnames(reference_data))], reference_data$labels[1:50])

get.knnx.prototype <- function(query_embedding, reference_embedding, knn = 30, k_prototypes = 5) {
  
  query_embedding <- matrix(query_embedding, nrow = 1)
  
  # find k-nearest neighbors and their prototyeps
  embedding <- reference_embedding[,grep('dim', colnames(reference_embedding))]
  labels <- reference_embedding$labels
  
  fnn_indicies <- get.knnx(embedding, query_embedding, k = knn)$nn.index
  prototypes <- get.prototypes(embedding[fnn_indicies, ], 
                               labels[fnn_indicies])
  
  if(ncol(prototypes) < k_prototypes) {
    k_prototypes <- min(k_prototypes, ncol(prototypes))
  }
  
  # find k-nearest prototypes
  knn_prototypes <- get.knnx(t(prototypes), query_embedding, k = k_prototypes)
  #list(nn.dist = knn_prototypes$nn.dist, nn.labels = rownames(t(prototypes))[knn_prototypes$nn.index] )
  
  probs <- exp(-0.5*((knn_prototypes$nn.dist^2)/sd(knn_prototypes$nn.dist)))
  probs <- probs/sum(probs)
  
  list(p =probs, nn.labels = rownames(t(prototypes))[knn_prototypes$nn.index] )

}


mappings_set1_above99 <- apply(query_data[query_data$Isotope.correlation > 0.99,
                                          grep('dim', colnames(query_data))],
                  1,
                  get.knnx.prototype,
                  reference_embedding = reference_data, knn = 30, k_prototypes = 5)





########################
#### second approach
####################

## step 1: screen prototypes
nnidx <- get.knnx(reference_data[,grep('dim', colnames(reference_data))], 
                  query_data[,grep('dim', colnames(query_data))], 
                  k = 30)

all_prototypes <- unique(reference_data$labels[as.numeric(nnidx$nn.index)])
length(all_prototypes)

get.prototypes <- function(latent, labels, prototype_labels){
  sapply(prototype_labels, FUN=function(x) colMeans(latent[labels == x,]))
  
}



embedding <- reference_data[,grep('dim', colnames(reference_data))]
labels <- reference_data$labels


prototypes <- get.prototypes(embedding, labels, all_prototypes)

 