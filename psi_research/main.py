def top_k_similar_users_with_recs(
    df,
    global_popularity,
    k=10,
    metric="euclidean",
    min_common=1,
    max_common=3,
    n_recommendations=2,
    atk=0
):
    if metric not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown metric '{metric}'")

    loss_fn = LOSS_FUNCTIONS[metric]
    users = df.index
    result = {}

    for ui in users:
        ui_ratings = df.loc[ui]
        distances = []

        # -------- FIND SIMILAR USERS --------
        for uj in users:
            if ui == uj:
                continue

            uj_ratings = df.loc[uj]

            common_mask = ui_ratings.notna() & uj_ratings.notna()
            num_common = common_mask.sum()

            if not (min_common <= num_common <= max_common):
                continue

            u = ui_ratings[common_mask].values
            v = uj_ratings[common_mask].values

            loss = loss_fn(u, v)
            distances.append((uj, loss))

        distances.sort(key=lambda x: x[1])
        top_neighbors = distances[:k]

        # -------- ATTACKER MODE --------
        if atk == 1:
            result[ui] = {
                "neighbors": top_neighbors
            }
            continue

        # -------- RECOMMEND NEW SERVICES --------
        service_scores = {}
        unrated_services = ui_ratings[ui_ratings.isna()].index

        for service in unrated_services:
            ratings = []

            for neighbor, _ in top_neighbors:
                r = df.loc[neighbor, service]
                if not np.isnan(r):
                    ratings.append(r)

            if ratings:
                mean_rating = np.mean(ratings)
                num_ratings = len(ratings)

                local_score = mean_rating * np.log(1 + num_ratings)

                service_scores[service] = {
                    "local_score": local_score,
                    "avg_rating": mean_rating,
                    "num_neighbors": num_ratings,
                    "global_popularity": global_popularity.get(service, 0.0)
                }

        top_services = sorted(
            service_scores.items(),
            key=lambda x: x[1]["local_score"],
            reverse=True
        )[:n_recommendations]

        # -------- STORE RESULT --------
        result[ui] = {
            "neighbors": top_neighbors,
            "recommendations": top_services
        }

    return result