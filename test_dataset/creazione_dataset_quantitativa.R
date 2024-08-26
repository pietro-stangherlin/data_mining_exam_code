# Creazione Dataset con dati mancanti 
# nota -> per ora non mi preoccupo della relazione con la y

rm(list = ls())

# numero di osservazioni
N = 10^3

# Il numero di variabili è relativamente contenuto
# l'obbiettivo è creare delle variabili esplicative con valori mancanti
# sia quantitative che qualitative
# più date con time stamp etc

df = data.frame(x1 = rep(NA, N))

# indici dei primi due terzi delle osservazioni
first_indexes = 1:round((NROW(df) * 2/3), 0)

# Per semplicità riempio sempre i primi due terzi di valori non mancanti

set.seed(123)

# Creazione delle variabili esplicative ------------------------------------


# x1 : NA + limitati valori numerici (qualitativa di fatto) ---------------
# qualitativa con valori mancanti

df$x1[first_indexes] = rbinom(length(first_indexes), 5, prob = 0.5)

head(df)


# x2 : NA + limitati valori stringhe ---------------
# qualitativa con valori mancanti

df$x2 = rep(NA, N)

df$x2[first_indexes] = sample(letters ,length(first_indexes), replace = T)

head(df)

# x3 : NA + limitati valori continui ---------------
# qualitativa con valori mancanti

df$x3 = rep(NA, N)

df$x3[first_indexes] = runif(length(first_indexes))

head(df)

# x4: NA + limitati valori stringhe + '' ----------------------

df$x4 = rep(NA, N)

df$x4[first_indexes] = sample(c(letters, '') ,length(first_indexes), replace = T)

head(df)

# x5: NA + molti valori stringhe + '' ----------------------
text_to_split = "Lorem ipsum dolor sit amet, 
consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation
ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit
in voluptate velit esse cillum dolore eu
fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae
vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut
odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi
nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet,
consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt
ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam,
quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea
commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit
esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur"

splitted_text = strsplit(text_to_split, split = " ")[[1]]


df$x5 = rep(NA, N)

df$x5[first_indexes] = sample(c(splitted_text, '') ,length(first_indexes), replace = T)

head(df)


# x6: NA + date sporche + '' ----------------------

# Generate a sequence of dates from January 1, 2022 to January 10, 2022
date_seq_base <- seq(
  as.Date("2022-01-01"), 
  as.Date("2025-01-10"), 
  by = "days")

head(date_seq_base)

date_seq_base = as.character(date_seq_base)

df$x6 = rep(NA, N)

df$x6[first_indexes] = sample(c(date_seq_base, '') ,length(first_indexes), replace = T)

head(df)

df$x6[first_indexes] = ifelse(df$x6[first_indexes] != '',
                              paste(df$x6[first_indexes],
                                    "41414141", sep = ""), df$x6[first_indexes])

head(df)

# aggiungo caratteri a destra di ogni stringa


# x7: NA + valori numerici ----------------------
df$x7 = rep(NA, N)

df$x7[first_indexes] = runif(length(first_indexes))

df$x7[first_indexes] = ifelse(df$x7[first_indexes] > 0.5, NA , df$x7[first_indexes])

head(df)


# x8: valori numerici non mancanti
df$x8 = runif(N)

# Aggiunta della y -------------

df$y = rnorm(N, df$x8 + ifelse(df$x2 == "a" | is.na(df$x2), 1, 0) + ifelse(!is.na(df$x7), df$x7, -2))

sum(is.na(df$x8))
sum(is.na(ifelse(df$x2 == "a" | is.na(df$x2), 1, 0)))


sum(is.na(df$x7))
sum(is.na(ifelse(!is.na(df$x7), df$x7, -2)))

# Scrittura file.csv --------------
write.csv(df, "df_quant.csv", row.names = TRUE, na = "")






