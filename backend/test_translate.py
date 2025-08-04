# test_translate.py

from translation_utils import translate_if_needed

samples = [
    "   申し訳ありませんが、特定の個人に関する情報を教えることはできません。個人のプライバシーを保護するため、具体的な情報の提供は制限されています。何か一般的な質問やトピックに関する情報をご希望でしたら、お手伝いできますので、ぜひお知らせください！",
    "Bonjour tout le monde",
    "你好，世界",
    "申し訳ありませんが、Arito Shimazakiという個人についての詳細な情報は持っていません。具体的な背景や目的に関する情報を教えていただければ、さらにお手伝いできるかと思います。もしこの人物が著名な方や公開情報に該当する場合、一般的に知られている情報ならサポートできる可能性があります"
]

for s in samples:
    out = translate_if_needed(s)
    print(f"{s!r} -> {out!r}")
