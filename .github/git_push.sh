if [[ `git status --porcelain` ]]; then
	git config --global user.email "lqs@mail.bnu.edu.cn"
	git config --global user.name "Qingsong Lv"
	git add README.md
	git commit -m "index file by github actions"
	git push origin docs
fi
