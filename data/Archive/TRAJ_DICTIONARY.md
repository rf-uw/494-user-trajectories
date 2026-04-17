# Data Dictionary

All datasets are aggregated per user per month. `userMonth` is the number of calendar months since the user's first action on Community Notes. Together the datasets contain the full trajectories for a sample of 20,000 users.

## Notes (`sample_user_note_traj.parquet`)

- `noteAuthorParticipantId` — User identifier
- `userMonth` — Calendar months since user's first action
- `calendarMonth` — Calendar month of activity (YYYY-MM)
- `notesCreated` — Number of notes written by user
- `hits` — Number of user's notes that achieved CRH (Currently Rated Helpful) status
- `hitRate` — Fraction of user's notes that achieved CRH status
- `avgNoteFactor` — Mean note factor of notes user wrote
- `avgNoteIntercept` — Mean note intercept of notes user wrote
- `topicsTargeted` — Number of distinct topics targeted
- `avgRatingsEarned` — Mean number of community ratings received per note
- `{topic}Count` — Notes written on each topic. Topics: `sports`, `diaries_&_daily_life`, `business_&_entrepreneurs`, `science_&_technology`, `news_&_social_concern`, `other`

## Ratings (`sample_user_rating_traj.parquet`)

- `raterParticipantId` — User identifier
- `userMonth` — Calendar months since user's first action
- `calendarMonth` — Calendar month of activity (YYYY-MM)
- `notesRated` — Total number of notes rated
- `avgHelpfulFactor`, `avgNotHelpfulFactor`, `helpfulNotHelpfulFactorDiff` — Mean note factor of notes the user rated helpful vs. not helpful
- `avgHelpfulIntercept`, `avgNotHelpfulIntercept`, `helpfulNotHelpfulInterceptDiff` — Mean note intercept of notes the user rated helpful vs. not helpful
- `correctHelpfuls` — Number of helpful ratings on notes that achieved CRH status
- `correctNotHelpfuls` — Number of non-helpful ratings on notes that did not achieve CRH status
- `overallAccuracy` — `(correctHelpfuls + correctNotHelpfuls) / notesRated`
- `posFactorRatedHelpful`, `posFactorRatedNotHelpful`, `negFactorRatedHelpful`, `negFactorRatedNotHelpful` — Count of ratings split by note factor sign (+/-) and whether the user rated it helpful or not
- `pctCorrectPosFactorHelpful`, `pctCorrectPosFactorNotHelpful`, `pctCorrectNegFactorHelpful`, `pctCorrectNegFactorNotHelpful` — Fraction correct within each factor-sign x helpfulness bucket
- `pctHelpfulRatingsCorrect` — Fraction of helpful ratings where the note achieved CRH
- `pctNotHelpfulRatingsCorrect` — Fraction of non-helpful ratings where the note did not achieve CRH
- `uniqueDaysRated` — Number of distinct days the user rated notes
- `avgPostsRatedPerDay` — `notesRated / uniqueDaysRated`
- `uniqueTopicsRated` — Number of distinct topics rated
- `{anti,pro}{Dem,Rep}{NN,NNN}Ratings` — Partisan rating classifications from Nudo et al. `anti`/`pro` = rating direction, `Dem`/`Rep` = post author party, `NN` = note claims misinformation, `NNN` = note claims not misinformation. 8 columns total
- `proDemRatings`, `antiDemRatings`, `proRepRatings`, `antiRepRatings` — Summed partisan totals across NN and NNN variants
- `{topic}RatedCount` — Notes rated per topic. Same 6 topics as the notes dataset

## Requests (`sample_user_request_traj.parquet`)

- `requesterParticipantId` — User identifier
- `userMonth` — Calendar months since user's first action
- `calendarMonth` — Calendar month of activity (YYYY-MM)
- `requestsMade` — Number of note requests submitted
- `numRequestsResultingInNote` — Requests where at least one note was written
- `numRequestsResultingInCrh` — Requests where at least one note achieved CRH status
- `pctRequestResultedInNote` — Fraction of requests resulting in a note
- `pctRequestResultedInCrh` — Fraction of requests resulting in a CRH note
