from veripet.research.taxonomy import BreedTaxonomy


def test_taxonomy_resolves_aliases():
    taxonomy = BreedTaxonomy.from_pairs(
        [
            ("border_collie", "Border collie"),
            ("border collie", "Border collie"),
            ("siberian_husky", "Siberian husky"),
        ]
    )
    assert taxonomy.normalize("border_collie") == "Border collie"
    assert taxonomy.normalize("Siberian Husky") == "Siberian husky"
    assert taxonomy.normalize(None) == "breed_unknown"
