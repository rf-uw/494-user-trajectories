_top_5_topics = ["sports", "diaries_&_daily_life", "business_&_entrepreneurs", "science_&_technology", "news_&_social_concern", "other"]
_null_means_0 = [
    "notesRequested", "notesRated", "notesWritten", 
    "correctNotHelpfuls", "correctHelpfuls", "hits", 
    "negFactorRatedNotHelpful", "posFactorRatedHelpful", "negFactorRatedHelpful", "posFactorRatedNotHelpful", 
    "uniqueDaysRated", "avgPostsRatedPerDay", "avgRatingsEarned",
    "uniqueTopicsRated", "uniqueTopicsTargeted", 
    "proRepRatings", "proDemRatings", "antiRepRatings", "antiDemRatings", 
    "repAlignedRatings", "demAlignedRatings", "ratingsOnReps", "ratingsOnDems", 
    "demAlignedLessRepAlignedRatings", 
     "proRepNotes", "proDemNotes", "antiRepNotes", "antiDemNotes", 
    "demAlignedNotes", "repAlignedNotes", "notesOnReps", "notesOnDems", 
    "demAlignedLessRepAlignedNotes", 
] + [f"{topic}NotesRated" for topic in _top_5_topics] + [f"{topic}NotesWritten" for topic in _top_5_topics]