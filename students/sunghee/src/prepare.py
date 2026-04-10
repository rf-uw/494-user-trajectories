# # src/prepare_polars.py

# NOTES_COLUMNS_TO_DROP = {
#     (2023, 10): ['isCollaborativeNote'],
#     (2023, 11): ['isCollaborativeNote'],
#     (2026, 1): ['isCollaborativeNote']
# }

# RATINGS_COLUMNS_TO_DROP = {
#     (2023, 11): ['ratingSourceBucketed'],
#     (2026, 1): ['ratingSourceBucketed']
# }

# STATUS_COLUMNS_TO_DROP = {
#     (2023, 11): [
#         'timestampMillisOfMostRecentStatusChange',
#         'timestampMillisOfNmrDueToMinStableCrhTime',
#         'currentMultiGroupStatus',
#         'currentModelingMultiGroup',
#         'timestampMinuteOfFinalScoringOutput',
#         'timestampMillisOfFirstNmrDueToMinStableCrhTime'],
#     (2023, 12): [],
#     (2026, 1): [
#         'timestampMillisOfMostRecentStatusChange',
#         'timestampMillisOfNmrDueToMinStableCrhTime',
#         'currentMultiGroupStatus',
#         'currentModelingMultiGroup',
#         'timestampMinuteOfFinalScoringOutput',
#         'timestampMillisOfFirstNmrDueToMinStableCrhTime'
#     ],
# }

# ENROLLMENT_COLUMNS_TO_DROP = {
#     (2023, 11): [],
#     (2023, 12): [],
#     (2026, 1): ['numberOfTimesEarnedOut'],
#     (2026, 2): ['numberOfTimesEarnedOut'],
# }


# def prepare_notes(df, year, month):
#     columns_to_drop = NOTES_COLUMNS_TO_DROP.get((year, month), [])
#     existing_cols_to_drop = [
#         col for col in columns_to_drop if col in df.columns
#     ]
#     if existing_cols_to_drop:
#         df = df.drop(existing_cols_to_drop)
#     return df


# def prepare_ratings(df, year, month):
#     columns_to_drop = RATINGS_COLUMNS_TO_DROP.get((year, month), [])
#     existing_cols_to_drop = [
#         col for col in columns_to_drop if col in df.columns
#     ]
#     if existing_cols_to_drop:
#         df = df.drop(existing_cols_to_drop)
#     return df


# def prepare_status(df, year, month):
#     columns_to_drop = STATUS_COLUMNS_TO_DROP.get((year, month), [])
#     existing_cols_to_drop = [
#         col for col in columns_to_drop if col in df.columns
#     ]
#     if existing_cols_to_drop:
#         df = df.drop(existing_cols_to_drop)
#     return df


# def prepare_enrollment(df, year, month):
#     columns_to_drop = ENROLLMENT_COLUMNS_TO_DROP.get((year, month), [])
#     existing_cols_to_drop = [
#         col for col in columns_to_drop if col in df.columns
#     ]
#     if existing_cols_to_drop:
#         df = df.drop(existing_cols_to_drop)
#     return df

def prepare_notes(df, constants):
    expected = constants.noteTSVColumns
    existing = [c for c in expected if c in df.columns]
    return df.select(existing)


def prepare_ratings(df, constants):
    expected = constants.ratingTSVColumns
    existing = [c for c in expected if c in df.columns]
    return df.select(existing)


def prepare_status(df, constants):
    expected = constants.noteStatusHistoryTSVColumns
    existing = [c for c in expected if c in df.columns]
    return df.select(existing)


def prepare_enrollment(df, constants):
    expected = constants.userEnrollmentTSVColumns
    existing = [c for c in expected if c in df.columns]
    return df.select(existing)
